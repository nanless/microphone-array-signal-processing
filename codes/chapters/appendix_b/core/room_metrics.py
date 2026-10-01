"""Real RIR metrics with explicit float64 support boundaries.

Ordinary dot products and fits retain their historical arithmetic. Dangerous
energy scales use dimensionless amplitudes/log energies; dangerous time axes
use a translated, span-normalized fit. These policies are not arbitrary-
precision solvers. A nonzero component lost by normalization is rejected.
"""
from __future__ import annotations
import math
import numpy as np


def real_array(value, name='input'):
    """Reject bool/string/object/complex before converting finite real arrays."""
    try:
        raw = np.asarray(value)
        if raw.dtype.kind not in 'iuf':
            raise ValueError(f'{name} must contain real numeric values, not bool')
        # Python mixed lists must not turn a bool into a silent integer.
        if isinstance(value, (list, tuple)):
            def visit(items):
                for item in items:
                    if isinstance(item, (list, tuple)):
                        visit(item)
                    elif isinstance(item, (bool, np.bool_)):
                        raise ValueError(f'{name} must not contain bool')
            visit(value)
        with np.errstate(over='ignore', invalid='ignore'):
            result = raw.astype(float)
    except (TypeError, OverflowError) as error:
        raise ValueError(f'{name} must be representable real numeric values') from error
    if not np.all(np.isfinite(result)):
        raise ValueError(f'{name} must be finite real values')
    if np.any((raw != 0) & (result == 0)):
        raise ValueError(f'{name} loses nonzero values in float64 conversion')
    return result


def _normalized(values):
    peak = float(np.max(np.abs(values)))
    if peak == 0:
        raise ValueError('energy requires a nonzero signal')
    reduced = values / peak
    if np.any((values != 0) & (reduced == 0)):
        raise ValueError('nonzero RIR components exceed normalized float64 support')
    squared = reduced * reduced
    if np.any((reduced != 0) & (squared == 0)):
        raise ValueError('nonzero RIR powers exceed normalized float64 support')
    return peak, reduced


def drr_db(full_rir, direct_rir):
    """Same-clock direct/reflection energy ratio in dB, not an output SNR.

    Finite positive component energies are required. Safe ordinary inputs use
    the original dot/ratio; dangerous scales use independent log energies.
    Difference overflow and lost nonzero normalized components are rejected.
    """
    full, direct = real_array(full_rir, 'full RIR'), real_array(direct_rir, 'direct RIR')
    if full.ndim != 1 or direct.ndim != 1 or not full.size or not direct.size:
        raise ValueError('RIRs must be nonempty one-dimensional arrays')
    length = max(full.size, direct.size)
    full = np.pad(full, (0, length-full.size))
    direct = np.pad(direct, (0, length-direct.size))
    with np.errstate(over='ignore', under='ignore', invalid='ignore', divide='ignore'):
        reflected = full-direct
        pd, pr = float(np.dot(direct, direct)), float(np.dot(reflected, reflected))
        ratio = np.float64(pd)/np.float64(pr)
    if not np.all(np.isfinite(reflected)):
        raise ValueError('reflected RIR difference exceeds float64 range')
    if not np.any(direct) or not np.any(reflected):
        raise ValueError('DRR needs nonzero direct and reflected energy')
    # Subnormal energies have reduced relative precision: do not divide them.
    tiny = np.finfo(float).tiny
    if (math.isfinite(pd) and math.isfinite(pr) and pd >= tiny and pr >= tiny
            and math.isfinite(float(ratio)) and ratio >= tiny):
        return 10.0*math.log10(float(ratio))
    sd, nd = _normalized(direct)
    sr, nr = _normalized(reflected)
    result = 10.0*((2*math.log10(sd)+math.log10(float(np.dot(nd, nd))))
                  -(2*math.log10(sr)+math.log10(float(np.dot(nr, nr)))))
    if not math.isfinite(result):
        raise ValueError('DRR exceeds finite float64 range')
    return result


def _fit_decay(times, decay):
    """Keep ordinary polynomial fit; condition exceptional absolute time axes."""
    span = float(times[-1]-times[0])
    if not math.isfinite(span) or span <= 0:
        raise ValueError('EDC time span must be positive and representable')
    ordinary = (1e-100 <= span <= 1e100 and abs(float(times[0])) <= 1e6*span)
    if ordinary:
        try:
            slope = float(np.polyfit(times, decay, 1)[0])
        except np.linalg.LinAlgError as error:
            raise ValueError('EDC fit is unsupported by float64') from error
        t20, t60 = -20.0/slope if slope else math.inf, -60.0/slope if slope else math.inf
    else:
        normalized = (times-times[0])/span
        if not np.all(np.isfinite(normalized)) or np.any(np.diff(normalized) <= 0):
            raise ValueError('EDC normalization loses distinct time points')
        slope_scaled = float(np.polyfit(normalized, decay, 1)[0])
        with np.errstate(over='ignore', under='ignore', divide='ignore'):
            slope = float(np.float64(slope_scaled)/span)
        t20, t60 = (-20.0/slope_scaled)*span, (-60.0/slope_scaled)*span
    if (not math.isfinite(slope) or slope >= 0 or not math.isfinite(t20)
            or not math.isfinite(t60) or t20 <= 0 or t60 <= 0):
        raise ValueError('EDC slope and positive decay times exceed float64 support')
    return {'slope_db_per_s': slope, 't20_s': t20, 't60_extrapolated_s': t60}


def t20_from_edc_points(times_s, decay_db):
    """Three or more preselected strictly falling -5..-25 dB points.

    This fixture fit is not a general RIR estimator or crossing interpolator.
    The slope and both returned positive times must be representable.
    """
    times, decay = real_array(times_s, 'EDC times'), real_array(decay_db, 'EDC dB')
    with np.errstate(over='ignore', invalid='ignore'):
        ordered = (np.all(np.diff(times) > 0) if times.ndim == 1 else False)
    if (times.ndim != 1 or decay.ndim != 1 or len(times) != len(decay)
            or len(times) < 3 or not ordered or np.any(np.diff(decay) >= 0)):
        raise ValueError('EDC points require three finite times and strictly falling dB')
    indices = np.flatnonzero((decay >= -25) & (decay <= -5))
    if len(indices) < 3 or decay[indices[0]] != -5 or decay[indices[-1]] != -25:
        raise ValueError('-5 to -25 dB fit interval is unavailable')
    return _fit_decay(times[indices], decay[indices])


def measured_t60_from_t20(rir, sample_rate=16000):
    """Noiseless finite RIR Schroeder -5..-25 dB fit; no interval shortening.

    Amplitude scaling is applied only if the historical square/cumsum route
    risks overflow, subnormal power or underflow. No noise-floor correction,
    truncation or measured-room validity is inferred from a returned T60.
    """
    impulse = real_array(rir, 'RIR')
    if impulse.ndim != 1 or impulse.size < 2:
        raise ValueError('RIR must be a finite one-dimensional vector')
    fs = real_array(sample_rate, 'sample rate')
    if fs.ndim != 0 or fs <= 0:
        raise ValueError('sample rate must be a positive real scalar')
    with np.errstate(over='ignore', under='ignore', invalid='ignore'):
        squares = impulse*impulse
        energy = np.cumsum(squares[::-1])[::-1]
    if not np.any(impulse):
        raise ValueError('RIR has no energy')
    if (not np.all(np.isfinite(energy)) or np.any((impulse != 0) & (squares < np.finfo(float).tiny))):
        _, normalized = _normalized(impulse)
        energy = np.cumsum(np.square(normalized[::-1]))[::-1]
    with np.errstate(divide='ignore', under='ignore'):
        ratio = energy/energy[0]
        decay = 10*np.log10(ratio)
    if np.any((energy != 0) & (ratio == 0)):
        raise ValueError('EDC normalization loses nonzero energy')
    start, stop = np.flatnonzero(decay <= -5), np.flatnonzero(decay <= -25)
    if not start.size or not stop.size or stop[0]-start[0] < 3:
        raise ValueError('RIR does not contain a usable -5 to -25 dB decay interval')
    times = np.arange(start[0], stop[0]+1, dtype=float)/float(fs)
    return _fit_decay(times, decay[start[0]:stop[0]+1])['t60_extrapolated_s']
