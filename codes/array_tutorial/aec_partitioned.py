"""Minimal partitioned-block frequency-domain adaptive echo canceller.

This is a real-valued, mono, block-synchronous teaching implementation of the
instantaneous-power candidate update in chapter 6, Eq. (6-3). It is not
SpeexDSP's AUMDF, WebRTC AEC3, a double-talk detector, or a real-time device
interface. Input calls contain whole ``block_length``-sample blocks; ``freeze``
is one externally supplied boolean decision per block.

The allowed ``0 <= step_size < 2`` interval is an input guard, not a global
stability theorem for colored references, double talk, or path changes.

The forward FFT uses NumPy's negative exponent; ``ifft`` carries the reciprocal
FFT length. Each partition stores ``2N`` frequency bins. Its first ``N``
inverse-FFT coefficients describe an FIR segment when ``constrain_gradient``
is enabled. The last segment's unused taps are also held at zero so a requested
filter length that is not divisible by ``N`` remains exact. The unconstrained
variant deliberately allows circular components and has no unique length-L FIR
coefficient vector.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


class PartitionedFDAFState:
    """Block-state PBFDAF with optional per-block time-domain projection.

    ``process`` returns prior residual and echo estimate, before that block's
    weight update. Processing an empty input leaves all state unchanged. An
    invalid or numerically overflowing call does not partially advance state.
    The implementation allocates intermediate arrays and is not thread-safe.
    """

    def __init__(
        self,
        filter_length: int,
        block_length: int,
        *,
        step_size: float = 0.5,
        epsilon: float = 1e-8,
        constrain_gradient: bool = True,
        initial_weights: np.ndarray | None = None,
    ) -> None:
        for name, value in (("filter_length", filter_length),
                            ("block_length", block_length)):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError(f"{name} must be a positive integer")
        if not isinstance(constrain_gradient, (bool, np.bool_)):
            raise ValueError("constrain_gradient must be boolean")
        step_size = finite_real_scalar(step_size, "step_size")
        epsilon = finite_real_scalar(epsilon, "epsilon")
        if not 0.0 <= step_size < 2.0:
            raise ValueError("step_size must satisfy 0 <= step_size < 2")
        if epsilon <= 0.0:
            raise ValueError("epsilon must be finite and positive")

        self.filter_length = int(filter_length)
        self.block_length = int(block_length)
        self.partitions = (self.filter_length + self.block_length - 1) // self.block_length
        self.fft_length = 2 * self.block_length
        self.step_size = step_size
        self.epsilon = epsilon
        self.constrain_gradient = bool(constrain_gradient)

        if initial_weights is None:
            initial = np.zeros(self.filter_length)
        else:
            initial = finite_real_array(initial_weights, "initial_weights")
            if initial.ndim != 1 or initial.shape != (self.filter_length,):
                raise ValueError("initial_weights must have shape (filter_length,)")
            initial = initial.copy()
        padded = np.pad(initial, (0, self.partitions * self.block_length
                                    - self.filter_length))
        partition_time = np.zeros((self.partitions, self.fft_length))
        partition_time[:, :self.block_length] = padded.reshape(
            self.partitions, self.block_length)
        self._initial_weights_fft = np.fft.fft(partition_time, axis=1)
        if not np.all(np.isfinite(self._initial_weights_fft)):
            raise ValueError("initial_weights FFT exceeds the float64 range")
        # A finite spectrum can still overflow inside inverse-FFT butterflies.
        # Reject such initial states before exposing a nonfinite FIR getter.
        try:
            with np.errstate(over="raise", invalid="raise"):
                recovered = np.fft.ifft(self._initial_weights_fft, axis=1)
        except FloatingPointError as error:
            raise ValueError("initial_weights FFT round trip exceeds the float64 range") from error
        if not np.all(np.isfinite(recovered)):
            raise ValueError("initial_weights FFT round trip exceeds the float64 range")
        self.reset()

    @property
    def partition_time_coefficients(self) -> np.ndarray:
        """Full ``P x 2N`` inverse-FFT coefficient array, returned as a copy."""

        try:
            with np.errstate(over="raise", invalid="raise"):
                time_coefficients = np.fft.ifft(self._weights_fft, axis=1)
        except FloatingPointError as error:
            raise ValueError("PBFDAF inverse FFT exceeds the float64 range") from error
        if not np.all(np.isfinite(time_coefficients)):
            raise ValueError("PBFDAF inverse FFT exceeds the float64 range")
        return time_coefficients.real.copy()

    @property
    def fir_weights(self) -> np.ndarray:
        """Return the exact length-L FIR only for the constrained variant."""

        if not self.constrain_gradient:
            raise ValueError("unconstrained circular state is not a length-L FIR")
        return self.partition_time_coefficients[:, :self.block_length].reshape(-1)[
            :self.filter_length].copy()

    @property
    def reference_overlap(self) -> np.ndarray:
        """Return the previous N reference samples, oldest first."""

        return self._reference_overlap.copy()

    @property
    def last_candidate_partition_time_coefficients(self) -> np.ndarray | None:
        """Pre-projection ``P x 2N`` taps for the last updated block, or None."""

        if self._last_candidate_time is None:
            return None
        return self._last_candidate_time.copy()

    def reset(self) -> None:
        """Restore constructor FIR, zero reference history, and diagnostics."""

        self._weights_fft = self._initial_weights_fft.copy()
        self._reference_overlap = np.zeros(self.block_length)
        self._reference_spectra = np.zeros((self.partitions, self.fft_length),
                                           dtype=complex)
        self._last_candidate_time: np.ndarray | None = None

    def process(
        self,
        reference: np.ndarray,
        microphone: np.ndarray,
        *,
        freeze: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Process whole blocks; return ``(prior_residual, prior_echo_hat)``.

        ``freeze[b]`` stops only block ``b``'s update. Its prediction and the
        reference-history shift still occur. A zero current block can therefore
        update an older partition when an earlier reference block was active.
        """

        x = finite_real_array(reference, "reference")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape:
            raise ValueError("reference and microphone must be equal-length 1-D arrays")
        if x.size % self.block_length:
            raise ValueError("inputs must contain whole block_length-sample blocks")
        nblocks = x.size // self.block_length
        if freeze is None:
            frozen = np.zeros(nblocks, dtype=bool)
        else:
            frozen = np.asarray(freeze)
            if frozen.dtype.kind != "b" or frozen.shape != (nblocks,):
                raise ValueError("freeze must be a boolean array with one value per block")
            frozen = frozen.copy()

        weights_fft = self._weights_fft.copy()
        overlap = self._reference_overlap.copy()
        spectra = self._reference_spectra.copy()
        last_candidate = self._last_candidate_time
        residual = np.empty_like(d)
        echo_hat = np.empty_like(d)
        n = self.block_length
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise",
                             under="ignore"):
                for b in range(nblocks):
                    sl = slice(b * n, (b + 1) * n)
                    block = x[sl]
                    spectrum = np.fft.fft(np.concatenate((overlap, block)))
                    spectra[1:] = spectra[:-1].copy()
                    spectra[0] = spectrum
                    if not np.all(np.isfinite(spectra)):
                        raise ValueError("reference FFT exceeds the float64 range")

                    prediction = np.sum(weights_fft * spectra, axis=0)
                    circular_output = np.fft.ifft(prediction)
                    echo_hat[sl] = circular_output[n:].real
                    residual[sl] = d[sl] - echo_hat[sl]
                    if not np.all(np.isfinite(residual[sl])):
                        raise ValueError("PBFDAF prediction exceeds the float64 range")

                    if not frozen[b] and self.step_size > 0.0:
                        error_fft = np.fft.fft(np.concatenate((np.zeros(n), residual[sl])))
                        power = np.sum(np.abs(spectra) ** 2, axis=0) + self.epsilon
                        candidate = weights_fft + (self.step_size * np.conj(spectra)
                                                   * error_fft / power)
                        if not np.all(np.isfinite(candidate)):
                            raise ValueError("PBFDAF update exceeds the float64 range")
                        candidate_time = np.fft.ifft(candidate, axis=1).real
                        last_candidate = candidate_time.copy()
                        if self.constrain_gradient:
                            candidate_time[:, n:] = 0.0
                            # Do not let padding in a short final partition
                            # change the requested length-L FIR.
                            remainder = self.filter_length - (self.partitions - 1) * n
                            candidate_time[-1, remainder:n] = 0.0
                            weights_fft = np.fft.fft(candidate_time, axis=1)
                        else:
                            weights_fft = candidate
                        if not np.all(np.isfinite(weights_fft)):
                            raise ValueError("PBFDAF projected update exceeds the float64 range")
                    else:
                        last_candidate = None
                    overlap = block.copy()
        except FloatingPointError as error:
            raise ValueError("PBFDAF intermediate exceeds the float64 range") from error

        self._weights_fft = weights_fft
        self._reference_overlap = overlap
        self._reference_spectra = spectra
        self._last_candidate_time = last_candidate
        return residual, echo_hat
