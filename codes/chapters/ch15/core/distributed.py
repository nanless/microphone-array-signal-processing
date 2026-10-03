"""Known-covariance linear estimation and scalar node-message controls.

Arrays use y=w.H@x and covariance E[x x.H]. Covariances are digital
amplitude squared; indices are zero based. A node estimates the target image
at its own explicitly selected microphone. These finite controls do not run
the MATLAB DANSE implementation, estimate covariances, or prove its theorem.
"""
from __future__ import annotations

import math
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_scalar
from codes.chapters.ch04.core.covariance import _validate_covariance, _load_covariance


def _complex(value, name):
    raw = np.asarray(value)
    if raw.dtype.kind not in 'iufc':
        raise ValueError(name+' must be numeric, not bool or text')
    with np.errstate(over='ignore', invalid='ignore'):
        out = raw.astype(complex)
    if not np.all(np.isfinite(out)):
        raise ValueError(name+' must be finite complex128')
    return out


def _integer(value, name, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(name+' must be an integer >= '+str(minimum))
    return int(value)


def _models(target_covariance, noise_covariance):
    rs = _validate_covariance(_complex(target_covariance, 'target covariance'))
    rn = _validate_covariance(_complex(noise_covariance, 'noise covariance'))
    if rs.shape != rn.shape:
        raise ValueError('target/noise covariance shapes differ')
    peak = float(max(np.max(np.maximum(abs(rs.real), abs(rs.imag))),
                     np.max(np.maximum(abs(rn.real), abs(rn.imag)))))
    if not peak:
        raise np.linalg.LinAlgError('zero observation covariance is not positive definite')
    # Divide components separately, including at subnormal powers.
    sn = rs.real/peak+1j*(rs.imag/peak)
    nn = rn.real/peak+1j*(rn.imag/peak)
    return rs, rn, sn, nn, peak


def _reference(reference, channels):
    reference = _integer(reference, 'reference')
    if reference >= channels:
        raise ValueError('reference is outside the global microphone array')
    return reference


def mwf_weights(target_covariance, noise_covariance, reference=0, *, condition_limit=1e12):
    """Solve (Rs+Rn)w=Rs[:,reference]; require the actual matrix to be SPD."""
    _, _, rs, rn, _ = _models(target_covariance, noise_covariance)
    reference = _reference(reference, len(rs))
    matrix = _load_covariance(rs+rn, 0., condition_limit)
    weights = np.linalg.solve(matrix, rs[:, reference])
    if not np.all(np.isfinite(weights)):
        raise ValueError('MWF weights exceed complex128 support')
    return weights


def _quadratic(matrix, vector, physical_scale):
    peak = float(np.max(np.maximum(abs(vector.real), abs(vector.imag))))
    if not peak:
        return 0.
    v = vector.real/peak+1j*(vector.imag/peak)
    reduced = float(np.vdot(v, matrix@v).real)
    # PSD validation permits small roundoff; a negative quadratic is not clipped.
    if reduced < 0:
        raise np.linalg.LinAlgError('quadratic power is negative; model is not usable at this direction')
    if reduced == 0:
        return 0.
    # Combine binary exponents before reconstruction. On some platforms
    # np.longdouble is float64, so a wider dtype cannot be assumed here.
    mantissa, exponent = 1., 0
    for factor in (reduced, peak, peak, physical_scale):
        part, power = math.frexp(factor)
        mantissa *= part; exponent += power
    try:
        result = math.ldexp(mantissa, exponent)
    except OverflowError as error:
        raise ValueError('nonzero physical power is outside float64 support') from error
    if not np.isfinite(result) or result == 0:
        raise ValueError('nonzero physical power is outside float64 support')
    return result


def mse_components(target_covariance, noise_covariance, weights, reference=0):
    """Complete cost for arbitrary w, including nonoptimal and zero weights.

Target and noise are assumed uncorrelated in this covariance model. The
distortion uses (w-e_r).H Rs (w-e_r), avoiding cancellation of large terms.
It is not a cost formula valid only at the MWF optimum.
"""
    _, _, rs, rn, scale = _models(target_covariance, noise_covariance)
    reference = _reference(reference, len(rs))
    w = _complex(weights, 'weights')
    if w.shape != (len(rs),):
        raise ValueError('weights must have one entry per global microphone')
    e = np.zeros(len(rs), complex); e[reference] = 1
    distortion = _quadratic(rs, w-e, scale)
    noise = _quadratic(rn, w, scale)
    total = distortion+noise
    if not np.isfinite(total):
        raise ValueError('total MSE exceeds float64 support')
    power = float(np.longdouble(rs[reference, reference].real)*np.longdouble(scale))
    if not np.isfinite(power) or (rs[reference, reference].real != 0 and power == 0):
        raise ValueError('reference target power is outside float64 support')
    normalized = total/power if power else None
    if normalized is not None and not np.isfinite(normalized):
        raise ValueError('normalized MSE exceeds float64 support')
    return {'target_distortion': distortion, 'noise_power': noise, 'total_mse': total,
            'reference_target_power': power, 'normalized_mse': normalized,
            'target_noise_cross_term': 0., 'cross_term_scope': 'uncorrelated covariance model'}


def _normalized_rows(projection, channels):
    t = _complex(projection, 'projection')
    if t.ndim != 2 or t.shape[1] != channels or not 1 <= t.shape[0] <= channels:
        raise ValueError('projection must have shape compressed channels x global channels, with d<=M')
    peak = np.max(np.maximum(abs(t.real), abs(t.imag)), axis=1)
    if np.any(peak == 0):
        raise np.linalg.LinAlgError('zero compression row must be rejected or explicitly removed')
    reduced = t.real/peak[:, None]+1j*(t.imag/peak[:, None])
    norms = np.sqrt(np.sum(abs(reduced)**2, axis=1))
    unit = reduced/norms[:, None]
    singular = np.linalg.svd(unit, compute_uv=False)
    if singular[-1] <= 1e-12*singular[0]:
        raise np.linalg.LinAlgError('compression rows are dependent or numerically indistinguishable')
    return unit, peak, norms


def compressed_mwf(target_covariance, noise_covariance, projection, reference=0, *, condition_limit=1e12):
    """LMMSE in explicitly normalized compression coordinates.

Rows describe observations z=T@x, so outgoing v.H@x uses a conjugated row.
Returned h applies to normalized_projection@x, not the original unnormalized
T@x. Row normalization is a coordinate change, not fitted output gain.
"""
    _, _, rs, rn, _ = _models(target_covariance, noise_covariance)
    reference = _reference(reference, len(rs))
    t, row_peaks, row_norms = _normalized_rows(projection, len(rs))
    matrix = t@(rs+rn)@t.conj().T
    matrix = _load_covariance(matrix, 0., condition_limit)
    p = t@rs[:, reference]
    h = np.linalg.solve(matrix, p)
    w = t.conj().T@h
    if not np.all(np.isfinite(w)) or not np.all(np.isfinite(h)):
        raise ValueError('compressed solve exceeds complex128 support')
    # Restore coefficients of the caller's real broadcast coordinates. Do not
    # form a potentially overflowing row norm or a complex reciprocal.
    reduced_h = h/row_norms
    raw_h = reduced_h.real/row_peaks+1j*(reduced_h.imag/row_peaks)
    if (not np.all(np.isfinite(raw_h))
            or np.any((reduced_h.real != 0) & (raw_h.real == 0))
            or np.any((reduced_h.imag != 0) & (raw_h.imag == 0))):
        raise ValueError('receiver coefficients in raw compression coordinates exceed complex128 support')
    return {'weights': w, 'compressed_weights': h, 'normalized_projection': t,
            'raw_compressed_weights': raw_h, 'raw_projection': _complex(projection, 'projection').copy(),
            'condition_number': float(np.linalg.cond(matrix)),
            'components': mse_components(target_covariance, noise_covariance, w, reference),
            'status': 'solved_in_normalized_coordinates'}


def known_models():
    a = np.array([1., .5, 2., -.5])
    white = np.eye(4)
    correlated = white.copy()
    correlated[0, 2] = correlated[2, 0] = .2
    correlated[0, 3] = correlated[3, 0] = .8
    return {'a': a, 'Rs': np.outer(a, a), 'white': white, 'correlated': correlated,
            'nodes': ((0, 1), (2, 3)), 'references': (0, 2)}


def _partition(nodes, channels, references):
    if isinstance(nodes, (str, bytes)) or len(nodes) < 2:
        raise ValueError('require at least two explicit node channel lists')
    nodes = tuple(tuple(_integer(j, 'channel index') for j in group) for group in nodes)
    flat = [j for group in nodes for j in group]
    if any(not group for group in nodes) or sorted(flat) != list(range(channels)):
        raise ValueError('nodes must partition all global channels exactly once')
    references = tuple(_reference(r, channels) for r in references)
    if len(references) != len(nodes) or any(r not in g for r, g in zip(references, nodes)):
        raise ValueError('each node reference must belong to that node')
    return nodes, references


def node_projection(nodes, compressions, node, channels, *, zero_policy='reject'):
    """Own raw channels followed by one normalized scalar per other node."""
    if zero_policy not in ('reject', 'local'):
        raise ValueError('zero_policy must be reject or local')
    own = np.eye(channels, dtype=complex)[list(nodes[node])]
    rows, missing = [row for row in own], []
    for peer, indices in enumerate(nodes):
        if peer == node:
            continue
        vector = _complex(compressions[peer], 'compression')
        if vector.shape != (len(indices),):
            raise ValueError('compression must match its node channel count')
        if not np.any(vector):
            if zero_policy == 'reject':
                raise np.linalg.LinAlgError('zero outgoing compression at peer '+str(peer))
            missing.append(peer); continue
        row = np.zeros(channels, complex); row[list(indices)] = vector.conj()
        rows.append(row)
    return np.array(rows), missing


def _receiver_projection(nodes, compressions, node, channels, peers):
    """Rebuild only the rows frozen into this receiver, without any solve."""
    rows = [row for row in np.eye(channels, dtype=complex)[list(nodes[node])]]
    for peer in peers:
        row = np.zeros(channels, complex)
        row[list(nodes[peer])] = compressions[peer].conj()
        rows.append(row)
    return np.array(rows)


def distributed_updates(target_covariance, noise_covariance, nodes, references, initial_compressions,
                        *, schedule='round_robin', relaxation=1., max_updates=40,
                        tolerance=1e-8, zero_policy='reject', condition_limit=1e12):
    """Finite known-covariance scalar-compression update control.

Initial output filters are actual local solves. A round-robin step solves
one node; a simultaneous epoch solves all nodes against the same old
compression snapshot. Relaxation only mixes the transmitted local weights;
    the stored solve proposal is undamped. Actual output after the broadcasts
    is reconstructed separately. Every history entry
    copies receiver coefficients and rebuilds all outputs under current
    broadcasts, including receivers with stale coefficients, without new
    solves. Old solve proposals are separately preserved. Stopping uses all
    current effective outputs' full normal-equation
residuals, an oracle diagnostic requiring the known global covariance.
It is neither a distributed covariance estimator nor a theorem guarantee.
"""
    rs_raw, rn_raw, rs, rn, _ = _models(target_covariance, noise_covariance)
    _load_covariance(rs+rn, 0., condition_limit)
    nodes, references = _partition(nodes, len(rs), references)
    relaxation = finite_real_scalar(relaxation, 'relaxation')
    tolerance = finite_real_scalar(tolerance, 'tolerance')
    max_updates = _integer(max_updates, 'max_updates', 1)
    if not 0 < relaxation <= 1 or tolerance < 0:
        raise ValueError('require 0<relaxation<=1 and tolerance>=0')
    if schedule not in ('round_robin', 'simultaneous'):
        raise ValueError('unknown update schedule')
    if len(initial_compressions) != len(nodes):
        raise ValueError('one initial compression is required per node')
    v = [_complex(c, 'initial compression').copy() for c in initial_compressions]
    for k, group in enumerate(nodes):
        if v[k].shape != (len(group),):
            raise ValueError('initial compression shape differs from node')
        node_projection(nodes, v, k, len(rs), zero_policy=zero_policy)
    outputs, output_time = [], [0]*len(nodes)
    receivers, receiver_peers, solve_outputs = [], [], []
    for k, group in enumerate(nodes):
        local = np.eye(len(rs), dtype=complex)[list(group)]
        initial = compressed_mwf(rs_raw, rn_raw, local, references[k], condition_limit=condition_limit)
        outputs.append(initial['weights'].copy())
        solve_outputs.append(initial['weights'].copy())
        receivers.append(initial['raw_compressed_weights'].copy())
        receiver_peers.append([])
    history = []; updates = 0; epoch = 0; status = 'budget_exhausted'
    visited = set()
    while updates < max_updates:
        active = [updates % len(nodes)] if schedule == 'round_robin' else list(range(len(nodes)))
        if updates+len(active) > max_updates:
            break  # Never make half of a simultaneous epoch appear simultaneous.
        before = [item.copy() for item in v]
        solutions = []
        for k in active:
            t, missing = node_projection(nodes, before, k, len(rs), zero_policy=zero_policy)
            solve = compressed_mwf(rs_raw, rn_raw, t, references[k], condition_limit=condition_limit)
            proposal = solve['weights'][list(nodes[k])]
            outgoing = (1-relaxation)*before[k]+relaxation*proposal
            if not np.all(np.isfinite(outgoing)):
                raise ValueError('outgoing relaxation exceeds complex128 support')
            solutions.append({'node': k, 'reference': references[k], 'incoming_compressions': [c.copy() for c in before],
                              'proposal_local_weights': proposal.copy(), 'new_compression': outgoing.copy(),
                              'missing_peers': missing,
                              'receiver_peers': [peer for peer in range(len(nodes)) if peer != k and peer not in missing],
                              **solve})
        epoch += 1; updates += len(active)
        for solution in solutions:
            k = solution['node']; v[k] = solution['new_compression'].copy()
            solve_outputs[k] = solution['weights'].copy()
            receivers[k] = solution['raw_compressed_weights'].copy()
            receiver_peers[k] = solution['receiver_peers'].copy()
            output_time[k] = updates; visited.add(k)
        # The receiver's coefficients can be stale while the remote signal is
        # newly compressed. Rebuild this actual linear operator, never reuse
        # an old effective full-array vector as if it were frozen physically.
        current_projections = []
        for k in range(len(nodes)):
            current_t = _receiver_projection(nodes, v, k, len(rs), receiver_peers[k])
            with np.errstate(over='ignore', invalid='ignore'):
                effective = current_t.conj().T@receivers[k]
            if not np.all(np.isfinite(effective)):
                raise ValueError('current-broadcast receiver output exceeds complex128 support')
            current_projections.append(current_t)
            outputs[k] = effective
        for solution in solutions:
            solution['effective_weights_after_broadcast'] = outputs[solution['node']].copy()
        residuals = []
        for w, reference in zip(outputs, references):
            p = rs[:, reference]; norm = float(np.linalg.norm(p))
            residual = float(np.linalg.norm((rs+rn)@w-p))
            residuals.append(residual/norm if norm else residual)
        history.append({'epoch': epoch, 'active_nodes': active, 'update_solve_count': updates,
                        'total_solve_count': len(nodes)+updates, 'solutions': solutions,
                        'compressions_before': before, 'compressions_after': [c.copy() for c in v],
                        'cached_outputs': [w.copy() for w in outputs], 'output_last_update': output_time.copy(),
                        'outputs_at_last_solve': [w.copy() for w in solve_outputs],
                        'receiver_coefficients_raw_coordinates': [u.copy() for u in receivers],
                        'receiver_peer_maps': [peers.copy() for peers in receiver_peers],
                        'current_receiver_projections': [t.copy() for t in current_projections],
                        'output_age_in_update_solves': [updates-t for t in output_time],
                        'cached_components': [mse_components(rs_raw, rn_raw, w, r) for w, r in zip(outputs, references)],
                        'normal_equation_relative_residuals': residuals})
        if len(visited) == len(nodes) and max(residuals) <= tolerance:
            status = 'known_covariance_residual_reached'; break
    return {'schedule': schedule, 'relaxation': relaxation, 'tolerance': tolerance,
            'max_update_solves': max_updates, 'update_solve_count': updates,
            'initial_local_solve_count': len(nodes), 'total_solve_count': len(nodes)+updates,
            'unused_update_budget': max_updates-updates, 'status': status,
            'compressions': [c.copy() for c in v], 'cached_outputs': [w.copy() for w in outputs],
            'receiver_coefficients_raw_coordinates': [u.copy() for u in receivers],
            'receiver_peer_maps': [peers.copy() for peers in receiver_peers],
            'history': history, 'stop_condition': 'all current-broadcast effective normal-equation residuals <= tolerance, after visiting every node',
            'scope': 'finite known-global-covariance teaching update; not MATLAB DANSE or its convergence proof'}


def run_covariance_experiments():
    model = known_models(); rs = model['Rs']; a = model['a']
    result = {'model': model, 'cases': {}}
    for name in ('white', 'correlated'):
        rn = model[name]; w = mwf_weights(rs, rn)
        local = compressed_mwf(rs, rn, np.eye(4)[:2])
        stale_t = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 2, -.5]])
        correct_t = stale_t.copy(); correct_t[2, 2:] = w[2:].real
        result['cases'][name] = {
            'central_weights': w, 'central_components': mse_components(rs, rn, w),
            'local': local, 'stale': compressed_mwf(rs, rn, stale_t),
            'correct': compressed_mwf(rs, rn, correct_t),
            'node2_weights': mwf_weights(rs, rn, 2),
            'node2_components': mse_components(rs, rn, mwf_weights(rs, rn, 2), 2)}
    args = (rs, model['correlated'], model['nodes'], model['references'], [a[:2], a[2:]])
    result['round_robin'] = distributed_updates(*args, max_updates=40)
    result['simultaneous'] = distributed_updates(*args, schedule='simultaneous', max_updates=100)
    result['simultaneous_half'] = distributed_updates(*args, schedule='simultaneous', relaxation=.5, max_updates=100)
    return result


def jacobi_control(relaxation=1., iterations=4):
    """Three-variable coupled Jacobi control, explicitly not a DANSE update."""
    alpha = finite_real_scalar(relaxation, 'relaxation')
    iterations = _integer(iterations, 'iterations', 1)
    if not 0 < alpha <= 1:
        raise ValueError('relaxation must lie in (0,1]')
    r = np.full((3, 3), .9); np.fill_diagonal(r, 1.)
    x = np.zeros(3); trace = [x.copy()]
    for _ in range(iterations):
        proposed = np.ones(3)-(r-np.eye(3))@x
        x = (1-alpha)*x+alpha*proposed
        if not np.all(np.isfinite(x)):
            raise ValueError('Jacobi iteration exceeds float64 support')
        trace.append(x.copy())
    iteration_matrix = (1-alpha)*np.eye(3)-alpha*(r-np.eye(3))
    return {'R': r, 'rhs': np.ones(3), 'relaxation': alpha, 'trace': np.array(trace),
            'solution': np.linalg.solve(r, np.ones(3)),
            'spectral_radius': float(max(abs(np.linalg.eigvalsh(iteration_matrix)))),
            'scope': 'coupled Jacobi linear-system control, not DANSE or a proof about it'}


def step_size_control(terms=8):
    terms = _integer(terms, 'terms', 1)
    if terms > 100000:
        raise ValueError('finite demonstration budget is 100000 terms')
    i = np.arange(terms)
    return {'terms': terms,
            'harmonic': {'finite_values': 1/(i+1), 'limit_zero': True, 'infinite_sum': 'diverges', 'infinite_square_sum': 'converges'},
            'constant_half': {'finite_values': np.full(terms, .5), 'limit_zero': False, 'infinite_sum': 'diverges', 'infinite_square_sum': 'diverges'},
            'geometric_half': {'finite_values': np.exp2(-i.astype(float)-1), 'limit_zero': True, 'infinite_sum': '1', 'infinite_square_sum': '1/3'},
            'scope': 'mathematical infinite-sum conditions are analytical statements, not inferred from these finite values'}


def tree_sum_control():
    """Real two-pass messages on 1--2--3; no TI-DANSE filter is implemented."""
    own = np.array([1., 2., 3.])
    message_32 = own[2]
    message_21 = own[1]+message_32
    total_1 = own[0]+message_21
    message_12 = total_1
    total_2 = message_12
    message_23 = total_2
    totals = np.array([total_1, total_2, message_23])
    self_only = np.array([own[1], own[0]+own[2], own[1]])
    return {'own': own, 'edge_messages': [{'from': 3, 'to': 2, 'value': message_32},
             {'from': 2, 'to': 1, 'value': message_21}, {'from': 1, 'to': 2, 'value': message_12},
             {'from': 2, 'to': 3, 'value': message_23}], 'totals': totals,
            'subtract_own': totals-own, 'neighbors_self_only': self_only,
            'scope': 'finite linear tree aggregation with a root and two passes; not a full TI-DANSE implementation'}


def gevd_control():
    """Independent whitening of a fixed 2x2 generalized eigenproblem."""
    rn = np.array([[1., .5], [.5, 1.]])
    a = np.array([1., 2.]); rs = np.outer(a, a); rx = rn+rs
    values, vectors = np.linalg.eigh(rn)
    whitener = (vectors*(1/np.sqrt(values)))@vectors.T
    lambdas, u = np.linalg.eigh(whitener@rx@whitener)
    order = np.argsort(lambdas)[::-1]; lambdas = lambdas[order]; u = u[:, order]
    generalized_vectors = whitener@u
    for column in range(2):
        j = np.argmax(abs(generalized_vectors[:, column]))
        if generalized_vectors[j, column] < 0:
            generalized_vectors[:, column] *= -1
    # Independent rank-one reconstruction uses only lambda_1-1, not lambda_1.
    direction = generalized_vectors[:, 0]
    reconstructed_target = (lambdas[0]-1)*np.outer(rn@direction, rn@direction)
    rank1_weights = np.linalg.solve(rn+reconstructed_target, reconstructed_target[:, 0])
    return {'Rn': rn, 'Rx': rx, 'a': a, 'whitener': whitener,
            'generalized_eigenvalues': lambdas, 'generalized_vectors': generalized_vectors,
            'generalized_residual': rx@generalized_vectors-rn@generalized_vectors@np.diag(lambdas),
            'noise_metric_gram': generalized_vectors.T@rn@generalized_vectors,
            'ordinary_eigenvalues': np.linalg.eigvalsh(rx)[::-1],
            'rank1_reconstructed_target': reconstructed_target, 'rank1_weights': rank1_weights,
            'components': mse_components(rs, rn, rank1_weights),
            'scope': 'fixed NumPy whitening control; not the upstream adaptive GEVD-DANSE chain'}
