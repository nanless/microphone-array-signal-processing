"""Two-band Haar examples that separate perfect reconstruction from AEC fit.

The unitary block Haar bank is deliberately tiny so every term can be checked
by hand. A one-sample physical echo delay becomes a 2x2 *cross-band* system,
even though the analysis/synthesis bank itself is exactly invertible. The
diagonal subband NLMS below is a distinct approximation, not the exact model
for an arbitrary time-domain FIR or a production oversampled filter bank.
"""

from __future__ import annotations

import numpy as np

from .conventions import finite_real_array, finite_real_scalar


def haar_analyze(signal: np.ndarray) -> np.ndarray:
    """Return shape ``(blocks, 2)``: low and high, no padding or delay."""

    values = finite_real_array(signal, "signal")
    if values.ndim != 1 or values.size % 2:
        raise ValueError("signal must be a 1-D array of even length")
    blocks = values.reshape(-1, 2)
    # Divide first to avoid a removable overflow in a+b near float64 max.
    with np.errstate(over="ignore", invalid="ignore"):
        transformed = np.column_stack((blocks[:, 0] / np.sqrt(2.0)
                                       + blocks[:, 1] / np.sqrt(2.0),
                                       blocks[:, 0] / np.sqrt(2.0)
                                       - blocks[:, 1] / np.sqrt(2.0)))
    if not np.all(np.isfinite(transformed)):
        raise ValueError("Haar analysis exceeds float64 range")
    return transformed


def haar_synthesize(bands: np.ndarray) -> np.ndarray:
    """Inverse of ``haar_analyze``; input shape is ``(blocks, 2)``."""

    values = finite_real_array(bands, "bands")
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("bands must have shape (blocks, 2)")
    out = np.empty((values.shape[0], 2), dtype=float)
    # Again divide first: low=high=1e308 has a finite reconstructed sample.
    out[:, 0] = values[:, 0] / np.sqrt(2.0) + values[:, 1] / np.sqrt(2.0)
    out[:, 1] = values[:, 0] / np.sqrt(2.0) - values[:, 1] / np.sqrt(2.0)
    if not np.all(np.isfinite(out)):
        raise ValueError("Haar synthesis exceeds float64 range")
    return out.ravel()


def two_tap_crossband_matrices(taps: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For ``y[n] = h0*x[n] + h1*x[n-1]``, return current/previous block matrices.

    With ``X_m = [low_m, high_m]``, exact subband output is
    ``Y_m = A0 @ X_m + A1 @ X_(m-1)`` and ``X_-1=0``. The off-diagonal terms
    are not an artifact of an imperfect filter bank: Haar is orthonormal.
    """

    h = finite_real_array(taps, "taps")
    if h.shape != (2,):
        raise ValueError("taps must have shape (2,)")
    h0, h1 = h
    a0 = np.array([[h0 + h1 / 2, h1 / 2],
                   [-h1 / 2, h0 - h1 / 2]], dtype=float)
    a1 = np.array([[h1 / 2, -h1 / 2],
                   [h1 / 2, -h1 / 2]], dtype=float)
    if not (np.all(np.isfinite(a0)) and np.all(np.isfinite(a1))):
        raise ValueError("crossband matrices exceed float64 range")
    return a0, a1


def two_tap_subband_outputs(
    reference: np.ndarray, taps: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(exact_bands, diagonal_only_bands, exact_time_output)``.

    Diagonal-only drops the off-diagonal entries of ``A0`` and ``A1``. It is
    *one particular* fixed approximation, not the optimum diagonal adaptive
    filter. This function does no adaptation.
    """

    x_bands = haar_analyze(reference)
    a0, a1 = two_tap_crossband_matrices(taps)
    previous = np.vstack((np.zeros((1, 2)), x_bands[:-1])) if x_bands.size else x_bands
    try:
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            exact = x_bands @ a0.T + previous @ a1.T
            diagonal = (x_bands @ np.diag(np.diag(a0))
                        + previous @ np.diag(np.diag(a1)))
            time_output = haar_synthesize(exact)
    except FloatingPointError as exc:
        raise ValueError("subband model exceeds float64 range") from exc
    if not (np.all(np.isfinite(exact)) and np.all(np.isfinite(diagonal))):
        raise ValueError("subband model exceeds float64 range")
    return exact, diagonal, time_output


class HaarDiagonalSubbandNLMSState:
    """Independent low/high real NLMS filters on nonoverlapping Haar blocks.

    ``num_taps`` counts *blocks* per band, not time-domain samples. This is
    a restricted model: no cross-band prediction. ``process`` takes complete
    pairs of time samples and an optional Boolean freeze mask per pair. The
    returned residual and prior echo estimate are synthesized in time order.
    The caller must align playback/microphone samples and handle double talk.
    """

    def __init__(self, num_taps: int, *, step_size: float = 0.5,
                 epsilon: float = 1e-8) -> None:
        if (isinstance(num_taps, (bool, np.bool_))
                or not isinstance(num_taps, (int, np.integer)) or num_taps <= 0):
            raise ValueError("num_taps must be a positive integer")
        mu = finite_real_scalar(step_size, "step_size")
        epsilon = finite_real_scalar(epsilon, "epsilon")
        if not 0.0 <= mu < 2.0 or epsilon <= 0.0:
            raise ValueError("require 0 <= step_size < 2 and epsilon > 0")
        self.num_taps = int(num_taps)
        self.step_size = mu
        self.epsilon = epsilon
        self.reset()

    @property
    def weights(self) -> np.ndarray:
        """Shape ``(2 bands, num_taps)``, current block first."""

        return self._weights.copy()

    @property
    def history(self) -> np.ndarray:
        """Shape ``(2 bands, num_taps-1)``, oldest block first."""

        return self._history.copy()

    def reset(self) -> None:
        self._weights = np.zeros((2, self.num_taps), dtype=float)
        self._history = np.zeros((2, self.num_taps - 1), dtype=float)

    def process(self, reference: np.ndarray, microphone: np.ndarray, *,
                freeze: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        x = finite_real_array(reference, "reference")
        d = finite_real_array(microphone, "microphone")
        if x.ndim != 1 or d.ndim != 1 or x.shape != d.shape or x.size % 2:
            raise ValueError("reference and microphone must be equal even-length 1-D arrays")
        blocks = x.size // 2
        if freeze is None:
            frozen = np.zeros(blocks, dtype=bool)
        else:
            frozen = np.asarray(freeze)
            if frozen.dtype.kind != "b" or frozen.shape != (blocks,):
                raise ValueError("freeze must be a Boolean array with one entry per two-sample block")

        x_bands = haar_analyze(x)
        d_bands = haar_analyze(d)
        joined = np.concatenate((self._history, x_bands.T), axis=1)
        weights = self._weights.copy()
        error_bands = np.empty_like(d_bands)
        prediction_bands = np.empty_like(d_bands)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise",
                             under="ignore"):
                for block in range(blocks):
                    for band in range(2):
                        u = joined[band, block:block + self.num_taps][::-1]
                        prediction = weights[band] @ u
                        error = d_bands[block, band] - prediction
                        prediction_bands[block, band] = prediction
                        error_bands[block, band] = error
                        if not frozen[block] and self.step_size and np.any(u):
                            denominator = u @ u + self.epsilon
                            if not (np.isfinite(denominator) and denominator > 0):
                                raise ValueError("subband NLMS normalization exceeds float64 range")
                            weights[band] += self.step_size * error * u / denominator
                errors = haar_synthesize(error_bands)
                predictions = haar_synthesize(prediction_bands)
        except FloatingPointError as exc:
            raise ValueError("subband NLMS intermediate exceeds float64 range") from exc
        if not (np.all(np.isfinite(weights)) and np.all(np.isfinite(errors))
                and np.all(np.isfinite(predictions))):
            raise ValueError("subband NLMS intermediate exceeds float64 range")
        self._weights = weights
        if self.num_taps > 1:
            self._history = joined[:, -(self.num_taps - 1):].copy()
        return errors, predictions
