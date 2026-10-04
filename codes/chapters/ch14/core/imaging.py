"""Small acoustic-imaging teaching kernels for extension I.

Positive-frequency convention: x[n] = Re(z exp(+j omega n)), output w^H x.
A and W have shape microphones x grid; P has shape scan x source. R is a
Hermitian mean-square CSM (digital amplitude squared unless calibrated).
These are original equation-level examples, not Acoular implementations.
No function imports data, writes files, or promises unique dense-grid inversion.
"""
from __future__ import annotations

from functools import wraps
from itertools import combinations
import numpy as np
from codes.chapters.ch02.core.conventions import (
    finite_real_array, finite_real_scalar, validate_positions, validate_cft,
    hermitian_part,
)


def _numeric(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                return function(*args, **kwargs)
        except (FloatingPointError, OverflowError) as exc:
            raise ValueError('calculation exceeds float64 support') from exc
    return wrapped


def _complex(value, name):
    original = np.asarray(value)
    if original.dtype.kind not in 'iufc':
        raise ValueError(name+' must be numeric, not bool or text')
    array = np.asarray(original, dtype=complex)
    if not np.all(np.isfinite(array)):
        raise ValueError(name+' must be finite')
    return array


def _matrix(value, name):
    array = _complex(value, name)
    if array.ndim != 2 or min(array.shape) < 1:
        raise ValueError(name+' must be a nonempty matrix')
    return array


def _complex_real_scale(array, scale):
    """Divide real/imaginary parts without forming a reciprocal of scale.

    NumPy complex division can overflow internally for a positive subnormal
    real divisor even when both normalized components are representable.
    """
    result = np.empty_like(array)
    result.real = array.real/scale
    result.imag = array.imag/scale
    return result


def _hermitian(value, *, positive=False):
    array = _matrix(value, 'CSM')
    if array.shape[0] != array.shape[1]:
        raise ValueError('CSM must be square')
    scale = float(np.max(abs(array)))
    if scale:
        normalized = _complex_real_scale(array, scale)
        if np.max(abs(normalized-normalized.conj().T)) > 1e-12:
            raise ValueError('CSM must be Hermitian')
        if positive and np.min(np.linalg.eigvalsh(hermitian_part(normalized))) < -1e-12:
            raise ValueError('full CSM must be positive semidefinite')
    # Exact Hermitian subnormals must not be halved into zero merely to
    # remove an asymmetry they do not have. Return a copy, never the input.
    if np.array_equal(array, array.conj().T):
        return array.copy()
    # Only the accepted relative round-off asymmetry is removed.
    return hermitian_part(array)


def _count(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


@_numeric
def spherical_steering(microphones, grid, frequency_hz, *, sound_speed=343., reference_position=None):
    """Return A relative to pressure at one reference position, not source watts.

For source j: A_mj=(r_ref,j/r_mj) exp(-jk(r_mj-r_ref,j)). All distances must
be nonzero. Changing the reference also changes each source's q.
"""
    microphones = validate_positions(microphones)
    grid = validate_positions(grid)
    if microphones.shape[1] != grid.shape[1]:
        raise ValueError('coordinate dimensions must agree')
    reference = (np.zeros(grid.shape[1]) if reference_position is None
                 else finite_real_array(reference_position, 'reference_position'))
    if reference.shape != (grid.shape[1],):
        raise ValueError('reference_position must have one coordinate vector')
    frequency = finite_real_scalar(frequency_hz, 'frequency_hz')
    speed = finite_real_scalar(sound_speed, 'sound_speed')
    if frequency < 0 or speed <= 0:
        raise ValueError('frequency must be nonnegative and sound speed positive')
    distances = np.linalg.norm(grid[:, None, :]-microphones[None, :, :], axis=2)
    reference_distances = np.linalg.norm(grid-reference, axis=1)
    if np.any(distances <= 0) or np.any(reference_distances <= 0):
        raise ValueError('source must not coincide with a microphone or reference')
    return (reference_distances[:, None]/distances * np.exp(
        -2j*np.pi*frequency/speed*(distances-reference_distances[:, None]))).T


@_numeric
def conventional_weights(steering):
    """Columnwise matched DS: W_j=A_j/(A_j^H A_j), unit target response."""
    a = _matrix(steering, 'steering')
    energies = np.sum(abs(a)**2, axis=0)
    if np.any(energies <= 0):
        raise ValueError('steering columns must have representable positive energy')
    return a/energies


@_numeric
def source_power_csm(steering, powers):
    """A diag(q) A^H, only for vanishing source cross-spectral terms."""
    a = _matrix(steering, 'steering')
    q = finite_real_array(powers, 'powers')
    if q.shape != (a.shape[1],) or np.any(q < 0):
        raise ValueError('one nonnegative power per source column is required')
    scaled = a*np.sqrt(q)
    result = scaled@scaled.conj().T
    if np.any(q > 0) and np.any(abs(a[:, q > 0]) > 0) and not np.any(abs(result) > 0):
        raise ValueError('nonzero CSM underflows float64')
    return result


@_numeric
def csm_from_amplitudes(amplitudes):
    """Average peak-amplitude outer products / 2, shape M x snapshots.

Use only interior positive-frequency real-signal tones. DC/Nyquist do not
obey this peak-amplitude factor; use the spectrum normalization below.
"""
    z = _matrix(amplitudes, 'amplitudes')
    scaled = z/np.sqrt(2*z.shape[1])
    result = scaled@scaled.conj().T
    # BLAS matmul need not honor NumPy's floating-point error mode.
    if not np.all(np.isfinite(result)):
        raise ValueError('calculation exceeds float64 support')
    if np.any(z != 0) and not np.any(result != 0):
        raise ValueError('nonzero CSM underflows float64')
    return result


@_numeric
def scan_power(csm, weights):
    """Return real w_j^H R w_j; retain negatives for indefinite controls."""
    r = _hermitian(csm)
    w = _matrix(weights, 'weights')
    if w.shape[0] != r.shape[0]:
        raise ValueError('CSM/weight channel count differs')
    return np.real(np.einsum('mg,mn,ng->g', w.conj(), r, w))


@_numeric
def point_spread_function(weights, steering):
    w, a = _matrix(weights, 'weights'), _matrix(steering, 'steering')
    if w.shape[0] != a.shape[0]:
        raise ValueError('weight/steering channel count differs')
    return abs(w.conj().T@a)**2


@_numeric
def damas_gauss_seidel(psf, dirty_map, *, iterations=100, relaxation=1., initial=None, sweep='forward'):
    """Nonnegative coordinate GS for Pq=b, not general least-squares NNLS.

One forward_backward iteration includes a complete forward AND backward
pass. Diagonal division is explicit; the unit-diagonal original helper is
a special case. No convergence guarantee is claimed for arbitrary PSFs.
"""
    p, b = finite_real_array(psf, 'PSF'), finite_real_array(dirty_map, 'dirty_map')
    if p.ndim != 2 or p.shape[0] != p.shape[1] or p.shape[0] < 1 or b.shape != (p.shape[0],):
        raise ValueError('square PSF and matching dirty vector are required')
    if np.any(p < 0) or np.any(np.diag(p) <= 0):
        raise ValueError('PSF must be nonnegative with positive diagonal')
    iterations = _count(iterations, 'iterations')
    relaxation = finite_real_scalar(relaxation, 'relaxation')
    if not 0 < relaxation <= 1 or sweep not in ('forward', 'forward_backward'):
        raise ValueError('require 0<relaxation<=1 and explicit sweep choice')
    q = np.zeros(len(b)) if initial is None else finite_real_array(initial, 'initial').copy()
    if q.shape != b.shape or np.any(q < 0):
        raise ValueError('initial must be a matching nonnegative vector')
    history, residuals = [q.copy()], [p@q-b]
    indices = list(range(len(b)))
    if sweep == 'forward_backward':
        indices += list(reversed(indices))
    for _ in range(iterations):
        for i in indices:
            off = p[i]@q-p[i, i]*q[i]
            q[i] = max(0., (1-relaxation)*q[i]+relaxation*(b[i]-off)/p[i, i])
        history.append(q.copy())
        residuals.append(p@q-b)
    return {'q': q, 'history': np.asarray(history), 'scan_residual_history': np.asarray(residuals),
            'iterations': iterations, 'passes_per_iteration': 1 if sweep == 'forward' else 2,
            'sweep': sweep, 'relaxation': relaxation}


@_numeric
def finite_nnls(matrix, target):
    """Exact active-set enumeration for <=8 variables; exponential teaching cost.

Each face is solved by NumPy LS. The returned residual and KKT violation
remain visible. This function deliberately rejects a 441-variable grid.
"""
    x, y = finite_real_array(matrix, 'matrix'), finite_real_array(target, 'target')
    if x.ndim != 2 or min(x.shape) < 1 or x.shape[1] > 8 or y.shape != (x.shape[0],):
        raise ValueError('finite NNLS supports nonempty matrices with 1..8 columns and matching targets')
    # Common scaling preserves the LS objective ordering and KKT zeros.
    scale = max(float(np.max(abs(x))), float(np.max(abs(y))))
    xs, ys = (x/scale, y/scale) if scale else (x.copy(), y.copy())
    best, best_score = np.zeros(x.shape[1]), float(ys@ys)
    best_set = ()
    for size in range(1, x.shape[1]+1):
        for active in combinations(range(x.shape[1]), size):
            coefficients = np.linalg.lstsq(xs[:, active], ys, rcond=None)[0]
            if np.any(coefficients < 0):
                continue
            candidate = np.zeros(x.shape[1]); candidate[list(active)] = coefficients
            residual = xs@candidate-ys
            score = float(residual@residual)
            if score < best_score:
                best, best_score, best_set = candidate, score, active
    residual = x@best-y
    gradient = xs.T@(xs@best-ys)
    positive = best > 0
    violation = max(float(np.max(abs(gradient[positive]))) if np.any(positive) else 0.,
                    float(np.max(np.maximum(-gradient[~positive], 0))) if np.any(~positive) else 0.)
    return {'q': best, 'residual': residual, 'squared_residual': float(residual@residual),
            'active_columns': list(best_set), 'scaled_kkt_violation': violation,
            'enumerated_faces': 2**x.shape[1], 'scope': 'finite active-set teaching LS, at most eight variables'}


@_numeric
def clean_sc_full_csm(csm, weights, *, iterations=20, damping=.6):
    """Author equation branch with full PSD CSM, not Acoular's fixed-point code.

P=w^H R w, h=Rw/P, G=P h h^H; subtract damping*G and allocate damping*P
at the peak grid index. The accumulated map is not a waveform or a proof
of true-source-power recovery. The input arrays are never modified.
"""
    r = _hermitian(csm, positive=True).copy()
    w = _matrix(weights, 'weights')
    if w.shape[0] != r.shape[0]:
        raise ValueError('CSM/weight channel count differs')
    iterations = _count(iterations, 'iterations')
    damping = finite_real_scalar(damping, 'damping')
    if not 0 < damping <= 1:
        raise ValueError('damping must be in (0,1]')
    cleaned, steps = np.zeros(w.shape[1]), []
    first_peak = float(np.max(scan_power(r, w)))
    tolerance = 64*np.finfo(float).eps*max(first_peak, 0.)
    for iteration in range(iterations):
        dirty = scan_power(r, w)
        index = int(np.argmax(dirty)); power = float(dirty[index])
        if power <= tolerance:
            break
        h = r@w[:, index]/power
        component = power*np.outer(h, h.conj())
        allocated = damping*power
        cleaned[index] += allocated
        r = hermitian_part(r-damping*component)
        steps.append({'iteration': iteration+1, 'peak_index': index, 'peak_power': power,
                      'allocated_power': allocated, 'h': h, 'component_csm': component,
                      'residual_scan': scan_power(r, w)})
    return {'clean_map': cleaned, 'residual_csm': r, 'steps': steps,
            'completed_iterations': len(steps), 'damping': damping,
            'relative_peak_stop_factor': 64*np.finfo(float).eps,
            'scope': 'full-CSM author-equation teaching branch; no diagonal removal or Acoular fixed-point loop'}


@_numeric
def hermitian_real_vector(matrix, *, frobenius=True):
    """Diagonals, upper real parts, upper imaginary parts; sqrt(2) off-diagonal.

With frobenius=True, vector norm squared equals the full matrix Frobenius
norm squared. False intentionally demonstrates unequal matrix weighting.
"""
    if type(frobenius) is not bool:
        raise ValueError('frobenius must be bool')
    r = _hermitian(matrix)
    i, j = np.triu_indices(len(r), 1)
    factor = np.sqrt(2.) if frobenius else 1.
    return np.concatenate([np.diag(r).real, factor*r[i, j].real, factor*r[i, j].imag])


@_numeric
def csm_residual(measured, model):
    """Physical squared errors plus a scale-stable relative Frobenius norm.

    Physical squares that round entirely to zero are retained and explicitly
    flagged. A genuinely zero reference has no relative error. Unsupported
    overflow raises ValueError; scaling does not invent a physical square.
    """
    measured, model = _hermitian(measured), _hermitian(model)
    if model.shape != measured.shape:
        raise ValueError('CSM shapes must agree')
    residual = hermitian_real_vector(model-measured)
    reference = hermitian_real_vector(measured)
    denominator = float(reference@reference)
    numerator = float(residual@residual)
    upper = hermitian_real_vector(model-measured, frobenius=False)
    upper_squared = float(np.sum(upper**2))
    scale = max(float(np.max(abs(measured))), float(np.max(abs(model))))
    relative = None
    if np.any(measured != 0):
        scaled_reference = hermitian_real_vector(_complex_real_scale(measured, scale))
        scaled_residual = hermitian_real_vector(
            _complex_real_scale(model, scale)-_complex_real_scale(measured, scale))
        # hypot avoids another square underflow when the two matrices have
        # very different magnitudes, even after the common normalization.
        reference_norm = np.hypot.reduce(abs(scaled_reference))
        residual_norm = np.hypot.reduce(abs(scaled_residual))
        relative = float(residual_norm/reference_norm)
    return {'frobenius_squared': numerator, 'reference_frobenius_squared': denominator,
            'relative_frobenius': relative,
            'upper_unweighted_squared': upper_squared,
            'squared_underflow': {
                'frobenius_squared': bool(numerator == 0 and np.any(residual != 0)),
                'reference_frobenius_squared': bool(denominator == 0 and np.any(reference != 0)),
                'upper_unweighted_squared': bool(upper_squared == 0 and np.any(upper != 0))}}


@_numeric
def single_source_csm_fit(csm, steering, *, include_white_noise=False):
    """Unconstrained LS for known rank-one dictionary, optionally sigma^2 I.

This is not the production CMF optimizer. A rank-deficient design returns
identifiable=False and no split parameters instead of inventing an answer.
"""
    if type(include_white_noise) is not bool:
        raise ValueError('include_white_noise must be bool')
    r = _hermitian(csm, positive=True)
    a = _complex(steering, 'steering')
    if a.shape != (len(r),) or not np.any(abs(a) > 0):
        raise ValueError('nonzero matching steering vector required')
    template = np.outer(a, a.conj())
    bases = [template]+([np.eye(len(r))] if include_white_noise else [])
    design = np.column_stack([hermitian_real_vector(t) for t in bases])
    target = hermitian_real_vector(r)
    solution, _, rank, _ = np.linalg.lstsq(design, target, rcond=None)
    identifiable = rank == len(bases)
    return {'identifiable': bool(identifiable), 'design_rank': int(rank),
            'q': float(solution[0]) if identifiable else None,
            'white_noise_variance': float(solution[1]) if identifiable and include_white_noise else None,
            'gram': design.T@design, 'rhs': design.T@target,
            'physical_nonnegative': bool(np.all(solution >= 0)) if identifiable else None,
            'scope': 'known single source, simplified unconstrained CSM Frobenius LS'}


@_numeric
def one_sided_csm_density(spectra, *, sample_rate_hz, n_fft, window_energy):
    """Welch-style mean CSM density from shared C x F x frames raw rFFT.

Interior bins factor 2; DC and even-length Nyquist factor 1. The odd-length
last bin remains an interior bin. Density integrates with df=fs/n_fft.
"""
    # Reject unexpected nonnumeric input before the shared CFT conversion.
    _complex(spectra, 'spectra')
    x = validate_cft(spectra)
    n_fft = _count(n_fft, 'n_fft', minimum=2)
    fs = finite_real_scalar(sample_rate_hz, 'sample_rate_hz')
    energy = finite_real_scalar(window_energy, 'window_energy')
    if fs <= 0 or energy <= 0 or x.shape[1] != n_fft//2+1:
        raise ValueError('positive fs/window energy and matching rFFT bins required')
    factors = np.full(x.shape[1], 2.); factors[0] = 1.
    if n_fft % 2 == 0:
        factors[-1] = 1.
    normalized = x/np.sqrt(fs)/np.sqrt(energy)/np.sqrt(x.shape[2])
    return np.einsum('mfl,nfl->fmn', normalized, normalized.conj())*factors[:, None, None]


@_numeric
def integrate_psd(density, bin_width_hz):
    density = finite_real_array(density, 'density')
    widths = finite_real_array(bin_width_hz, 'bin_width_hz')
    if density.ndim < 1 or density.shape[0] < 1 or np.any(density < 0):
        raise ValueError('nonnegative density with leading frequency axis required')
    if widths.ndim == 0:
        widths = np.full(density.shape[0], float(widths))
    if widths.shape != (density.shape[0],) or np.any(widths <= 0):
        raise ValueError('one positive integration width per frequency bin required')
    return np.tensordot(widths, density, axes=(0, 0))


@_numeric
def region_power(values, mask, *, cell_area_m2=None):
    """Sum point quantities, or integrate declared area density with cell areas."""
    values = finite_real_array(values, 'values')
    selection = np.asarray(mask)
    if values.ndim != 1 or selection.dtype.kind != 'b' or selection.shape != values.shape:
        raise ValueError('one-dimensional values and matching boolean mask required')
    if cell_area_m2 is None:
        return float(np.sum(values[selection]))
    areas = finite_real_array(cell_area_m2, 'cell_area_m2')
    if areas.ndim == 0:
        areas = np.full(values.size, float(areas))
    if areas.shape != values.shape or np.any(areas <= 0):
        raise ValueError('positive scalar or per-cell areas required for area density')
    return float(np.sum(values[selection]*areas[selection]))


def two_cell_experiment():
    """Matched, coherent and partial-coherence controls; NumPy arrays for plots."""
    a = np.array([[1., 1.], [1., np.exp(-2j*np.pi/3)]], complex)
    w = conventional_weights(a); p = point_spread_function(w, a)
    q = np.array([1., .25]); independent = source_power_csm(a, q)
    v = a@np.array([1., .5]); coherent = np.outer(v, v.conj())
    cases = {}
    for name, r in [('independent', independent), ('coherent', coherent)]:
        b = scan_power(r, w); estimate = np.linalg.solve(p, b)
        cases[name] = {'R': r, 'b': b, 'q_inverse': estimate, 'gs': damas_gauss_seidel(p, b),
                       'full_csm_residual': csm_residual(r, source_power_csm(a, estimate))}
    gammas = np.linspace(-1., 1., 41)
    estimates = []
    for gamma in gammas:
        s = np.array([[1., .5*gamma], [.5*gamma, .25]])
        estimates.append(np.linalg.solve(p, scan_power(a@s@a.conj().T, w)))
    return {'A': a, 'W': w, 'P': p, 'q': q, 'cases': cases,
            'coherence_gamma': gammas, 'coherence_inverse_q': np.asarray(estimates),
            'scope': 'two known grid cells; digital mean-square reference, not acoustic watts'}


def damas_csm_objective_experiment():
    """E14-17: full CSM Gram objective versus scan-domain squared residual.

    Only matched conventional weights and unchanged full rank-one templates
    are used. Both minimizers are model fits, not actual source recovery.
    The existing finite NNLS and coordinate GS kernels remain unique.
    """
    a = np.array([[1., 1.], [1., np.exp(-2j*np.pi/3)]], complex)
    w = conventional_weights(a)
    v = np.sqrt(4/3)*np.array([1., np.exp(1j*np.pi/3)])
    r = np.outer(v, v.conj())
    p, b = point_spread_function(w, a), scan_power(r, w)
    templates = [np.outer(a[:, j], a[:, j].conj()) for j in range(2)]
    design = np.column_stack([hermitian_real_vector(t) for t in templates])
    target = hermitian_real_vector(r)
    gram, rhs = design.T@design, design.T@target
    d = np.sum(abs(a)**2, axis=0)**2
    gs = damas_gauss_seidel(p, b, iterations=10)
    scan = finite_nnls(p, b)
    csm = finite_nnls(design, target)
    common_scale = max(float(np.max(abs(design))), float(np.max(abs(target))))
    losses, scaled_losses = {}, {}
    for label, q in [('GS', gs['q']), ('scan', scan['q']), ('CSM', csm['q'])]:
        residual = design@q-target
        losses[label] = {'scan_squared': float(np.sum((p@q-b)**2)),
                         'csm_frobenius_squared': float(residual@residual)}
        scaled = (design/common_scale)@q-target/common_scale
        scaled_losses[label] = float(scaled@scaled)
    unequal = a*np.array([1., 2.])
    unequal_w = conventional_weights(unequal)
    unequal_design = np.column_stack([
        hermitian_real_vector(np.outer(column, column.conj()))
        for column in unequal.T])
    return {'A': a, 'W': w, 'R': r, 'P': p, 'b': b, 'D': np.diag(d),
            'G': gram, 'h': rhs, 'csm_dictionary_real': design,
            'csm_target_real': target, 'q_GS': gs['q'], 'q_scan': scan['q'],
            'q_CSM': csm['q'], 'losses': losses,
            'csm_gradient_at_solution': gram@csm['q']-rhs,
            'objective_common_scale': common_scale, 'scaled_csm_objectives': scaled_losses,
            'nonuniform_control': {
                'A': unequal, 'W': unequal_w,
                'P': point_spread_function(unequal_w, unequal),
                'D': np.diag(np.sum(abs(unequal)**2, axis=0)**2),
                'G': unequal_design.T@unequal_design,
                'b': scan_power(r, unequal_w), 'h': unequal_design.T@target},
            'scope': 'full CSM Frobenius with matched conventional weights; no DR, noise column, trace budget or intercept'}


def distinct_column_ambiguity_experiment():
    """E14-18: distinct columns, yet a feasible collective null direction."""
    a = np.array([[1., 1., 1., 1.], [1., 1j, -1., -1j]], complex)
    w = conventional_weights(a)
    design = np.column_stack([hermitian_real_vector(np.outer(column, column.conj()))
                              for column in a.T])
    examples = np.array([[1., 0., 1., 0.], [0., 1., 0., 1.], [.5, .5, .5, .5]])
    matrices = np.asarray([source_power_csm(a, q) for q in examples])
    p = point_spread_function(w, a)
    return {'A': a, 'W': w, 'P': p, 'R': matrices[0],
            'csm_dictionary_real': design, 'dictionary_rank': int(np.linalg.matrix_rank(design)),
            'psf_rank': int(np.linalg.matrix_rank(p)), 'psf_eigenvalues': np.linalg.eigvalsh(p),
            'null_direction': np.array([1., -1., 1., -1.]),
            'q_examples': examples, 'model_csms': matrices,
            'b_examples': np.asarray([scan_power(r, w) for r in matrices]),
            'feasible_family_t_interval': [-.5, .5],
            'alternative_white_noise': {'q': np.zeros(4), 'variance': 2., 'R': 2*np.eye(2)},
            'scope': 'first assume known zero sensor noise; unknown white noise is a separate enlarged model'}


def spherical_scan_experiment():
    """Actual 8-microphone 441-grid free-field scan, no dense inversion claim."""
    angles = 2*np.pi*np.arange(8)/8
    microphones = np.column_stack([.2*np.cos(angles), .2*np.sin(angles), np.zeros(8)])
    axis = np.linspace(-.3, .3, 21)
    x, y = np.meshgrid(axis, axis, indexing='xy')
    grid = np.column_stack([x.ravel(), y.ravel(), np.full(x.size, .6)])
    a = spherical_steering(microphones, grid, 4000.)
    w = conventional_weights(a)
    indices = np.array([215, 225]); q = np.array([1., .25])
    r = source_power_csm(a[:, indices], q)
    columns = point_spread_function(w, a[:, indices])
    b = scan_power(r, w)
    directions = grid/np.linalg.norm(grid, axis=1)[:, None]
    plane_a = np.exp(2j*np.pi*4000/343*(microphones@directions.T))
    plane_b = scan_power(r, conventional_weights(plane_a))
    peaks = []
    for source, source_index in enumerate(indices):
        local = np.flatnonzero(np.linalg.norm(grid[:, :2]-grid[source_index, :2], axis=1) <= .09+1e-12)
        peak = int(local[np.argmax(b[local])]); plane_peak = int(local[np.argmax(plane_b[local])])
        peaks.append({'source_index': source, 'true_grid_index': int(source_index), 'peak_grid_index': peak,
                      'true_position_m': grid[source_index], 'peak_position_m': grid[peak],
                      'peak_power': float(b[peak]), 'position_error_m': float(np.linalg.norm(grid[peak]-grid[source_index])),
                      'plane_peak_position_m': grid[plane_peak], 'plane_peak_power': float(plane_b[plane_peak])})
    changed_a = spherical_steering(microphones, grid[indices], 4000., reference_position=[0, 0, .2])
    old_distance = np.linalg.norm(grid[indices], axis=1)
    new_distance = np.linalg.norm(grid[indices]-[0, 0, .2], axis=1)
    changed_q = q*(old_distance/new_distance)**2
    changed_r = source_power_csm(changed_a, changed_q)
    return {'microphones_m': microphones, 'grid_m': grid, 'grid_axis_m': axis, 'grid_shape': [21, 21],
            'frequency_hz': 4000., 'sound_speed_m_s': 343., 'reference_position_m': [0., 0., 0.],
            'A': a, 'W': w, 'source_grid_indices': indices, 'source_positions_m': grid[indices],
            'q': q, 'R': r, 'psf_columns': columns, 'b': b, 'plane_mismatch_b': plane_b,
            'local_peaks': peaks, 'dirty_grid_sum': float(b.sum()), 'source_power_sum': float(q.sum()),
            'reference_change': {'reference_position_m': [0., 0., .2], 'q': changed_q,
                                 'R': changed_r, 'max_csm_difference': float(np.max(abs(changed_r-r)))},
            'scope': 'exact spherical monopoles at z=.6 m; grid-cell reference pressure, no unique 441-variable inverse'}
