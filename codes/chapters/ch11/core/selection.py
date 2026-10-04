"""Small deterministic selection helpers; no device performance is inferred.

Intervals are assumed valid bounds, not automatically confidence intervals.
Token scoring is exhaustive teaching enumeration (at most 4 streams, 6 reference
utterances, 24 tokens per side), not a replacement for a production WER scorer.
"""
from itertools import permutations, product
from collections.abc import Sequence
from fractions import Fraction
import math
from numbers import Integral
import numpy as np


def _comparison_value(value):
    """Keep integer order exact, including mixed integer/float comparisons.

    Python compares int to float without first rounding the integer. Restrict
    non-integral inputs to ordinary <=64-bit floating scalars; strings, complex,
    booleans and extended-precision values must not be silently coerced.
    """
    if isinstance(value, (bool, np.bool_)):
        raise ValueError('comparison values must not be booleans')
    if isinstance(value, Integral):
        return int(value)
    if not isinstance(value, (float, np.floating)):
        raise ValueError('comparison values must be integers or floating scalars')
    if isinstance(value, np.floating) and value.dtype.itemsize > 8:
        raise ValueError('extended-precision floating values are not supported')
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('comparison values must be finite')
    return value



def _sequence(value, name):
    if isinstance(value, (str, bytes, bytearray)) or not (
            isinstance(value, Sequence) or isinstance(value, np.ndarray) and value.ndim >= 1):
        raise ValueError(f'{name} must be an explicit ordered sequence')
    return list(value)


def _tokens(value, name):
    result = _sequence(value, name)
    if any(not isinstance(token, str) for token in result):
        raise ValueError(f'{name} tokens must be strings')
    return result


def _bounds(value):
    bounds = _sequence(value, 'interval')
    if len(bounds) != 2:
        raise ValueError('an interval needs lower and upper bounds')
    lower, upper = [_comparison_value(v) for v in bounds]
    if lower > upper:
        raise ValueError('lower must not exceed upper')
    return lower, upper


def upper_limit_verdict(intervals, limits):
    """All metrics are minimized; None means unmeasured, equality is allowed."""
    intervals = _sequence(intervals, 'intervals')
    limits = _sequence(limits, 'limits')
    if len(intervals) != len(limits) or len(limits) == 0:
        raise ValueError('provide equally sized nonempty intervals and limits')
    statuses = []
    for interval, limit in zip(intervals, limits):
        limit = _comparison_value(limit)
        if interval is None:
            statuses.append('undetermined')
            continue
        lower, upper = _bounds(interval)
        statuses.append('fail' if lower > limit else
                        'pass' if upper <= limit else 'undetermined')
    return ('fail' if 'fail' in statuses else 'undetermined'
            if 'undetermined' in statuses else 'pass')


def pareto_minima(costs):
    """Indices not strictly dominated: all costs <= and at least one <."""
    # dtype=object is required at the first conversion: a mixed list may lose
    # large integer bits before validation if NumPy infers float64 here.
    raw = np.asarray(costs, dtype=object)
    if raw.ndim != 2 or min(raw.shape) == 0:
        raise ValueError('costs must be a finite nonempty matrix')
    matrix = [[_comparison_value(value) for value in row] for row in raw]
    return [i for i, row in enumerate(matrix)
            if not any(all(a <= b for a, b in zip(other, row))
                       and any(a < b for a, b in zip(other, row))
                       for other in matrix)]


def token_edit_distance(reference, hypothesis):
    """Unit-cost token insertions, deletions and substitutions."""
    reference = _tokens(reference, 'reference')
    hypothesis = _tokens(hypothesis, 'hypothesis')
    row = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        previous, row = row, [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j-1] + (token != other)))
    return row[-1]


def score_session_events(sessions, *, scorer=token_edit_distance):
    """Keep planned sessions, valid empty output and scoring failures separate.

    References are explicit nonempty token lists. An absent ``hypothesis`` key
    means missing output; an empty token list is a valid hypothesis. The
    callback supplies only a total edit count, so this wrapper never infers
    substitutions/deletions/insertions or fabricates an edit traceback.
    Complete-set WER is undefined if any planned session fails to score.
    """
    sessions = _sequence(sessions, 'sessions')
    if not 1 <= len(sessions) <= 64 or not callable(scorer):
        raise ValueError('one to 64 planned sessions and a callable scorer required')
    prepared, seen = [], set()
    for session in sessions:
        if not isinstance(session, dict):
            raise ValueError('each planned session must be a dictionary')
        identifier = session.get('session_id')
        if not isinstance(identifier, str) or not identifier or identifier in seen:
            raise ValueError('planned session IDs must be unique nonempty strings')
        reference = _tokens(session.get('reference'), 'reference')
        if not reference:
            raise ValueError('each planned reference must contain at least one token')
        seen.add(identifier)
        prepared.append((session, reference))
    rows, words, errors = [], 0, 0
    for session, reference in prepared:
        row = {'session_id': session['session_id'], 'reference_words': len(reference),
               'errors': None, 'wer': None, 'exception_type': None,
               'exception_message': None}
        if 'hypothesis' not in session:
            row['status'] = 'missing_hypothesis'
        else:
            try:
                hypothesis = _tokens(session['hypothesis'], 'hypothesis')
            except ValueError as error:
                row.update(status='format_error', exception_type=type(error).__name__,
                           exception_message=str(error))
            else:
                try:
                    # A callback may alter its own token lists. Preserve the
                    # frozen plan and coverage denominators independently.
                    count = scorer(list(reference), list(hypothesis))
                    if isinstance(count, (bool, np.bool_)) or not isinstance(count, Integral) or count < 0:
                        raise ValueError('scorer must return a nonnegative integer edit count')
                except Exception as error:
                    row.update(status='scoring_exception', exception_type=type(error).__name__,
                               exception_message=str(error))
                else:
                    count = int(count)
                    row.update(status='empty_output' if not hypothesis else 'scored',
                               errors=count, wer=count / len(reference))
                    words += len(reference)
                    errors += count
        rows.append(row)
    successful = sum(row['errors'] is not None for row in rows)
    planned_words = sum(len(reference) for _, reference in prepared)
    complete = successful == len(rows)
    return {'rows': rows, 'planned_sessions': len(rows), 'scored_sessions': successful,
            'failed_sessions': len(rows)-successful, 'planned_reference_words': planned_words,
            'scored_reference_words': words, 'scored_errors': errors,
            'session_coverage': successful/len(rows), 'reference_word_coverage': words/planned_words,
            'successful_subset_wer': errors/words if words else None,
            'complete_wer': errors/planned_words if complete else None,
            'eligible_for_complete_ranking': complete,
            'failure_policy': 'retain unscored sessions; missing hypotheses are not imputed as empty',
            'edit_components_inferred': False}


def small_slot_word_errors(utterances, speakers, hypotheses):
    """Compare global speaker permutation with utterance-to-slot assignment.

    utterances are ordered token lists, with matching integer speaker labels.
    No timestamps or time-constrained matching; reference order is supplied by
    the caller. ORC concatenates assigned utterances in that order. cp pads
    empty streams to the larger speaker/slot count. Missing words are deletions.
    """
    utterances = [_tokens(seq, 'utterance') for seq in _sequence(utterances, 'utterances')]
    hypotheses = [_tokens(seq, 'hypothesis') for seq in _sequence(hypotheses, 'hypotheses')]
    speakers = _sequence(speakers, 'speakers')
    if not 1 <= len(utterances) <= 6 or len(speakers) != len(utterances):
        raise ValueError('one to six labeled utterances required')
    if not 1 <= len(hypotheses) <= 4:
        raise ValueError('one to four hypothesis slots required')
    if any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
           or v < 0 for v in speakers):
        raise ValueError('speaker labels must be nonnegative integers')
    if any(not isinstance(token, str) for seq in [*utterances, *hypotheses] for token in seq):
        raise ValueError('tokens must be strings')
    words = sum(map(len, utterances))
    if not 1 <= words <= 24 or sum(map(len, hypotheses)) > 24:
        raise ValueError('one to 24 reference words and at most 24 hypothesis words')
    labels = sorted(set(speakers))
    if len(labels) > 4:
        raise ValueError('at most four reference speakers')
    references = [[t for u, speaker in zip(utterances, speakers) if speaker == label
                   for t in u] for label in labels]
    count = max(len(references), len(hypotheses))
    references += [[] for _ in range(count - len(references))]
    padded = list(hypotheses) + [[] for _ in range(count - len(hypotheses))]
    cp = min(sum(token_edit_distance(ref, padded[j]) for ref, j in zip(references, order))
             for order in permutations(range(count)))
    orc, best = math.inf, None
    for assignment in product(range(len(hypotheses)), repeat=len(utterances)):
        concatenated = [[token for u, slot in zip(utterances, assignment) if slot == j
                         for token in u] for j in range(len(hypotheses))]
        error = sum(token_edit_distance(ref, hyp) for ref, hyp in zip(concatenated, hypotheses))
        if error < orc:
            orc, best = error, assignment
    return {'reference_words': words, 'cp_errors': cp, 'orc_errors': orc,
            'cp_wer': cp / words, 'orc_wer': orc / words,
            'orc_assignment_utterance_to_slot': list(best)}


def interval_dominates(a, b):
    """Certain dominance for minimized metrics, not dominance of midpoints.

    Every upper bound of A must be <= the corresponding lower bound of B,
    with at least one strict inequality. None means unknown and prevents a
    positive claim; all provided bounds are validated even when unknown.
    """
    a, b = _sequence(a, 'a'), _sequence(b, 'b')
    if not a or len(a) != len(b):
        raise ValueError('equally sized nonempty metric intervals required')
    left = [None if v is None else _bounds(v) for v in a]
    right = [None if v is None else _bounds(v) for v in b]
    if any(v is None for v in left + right):
        return False
    return all(x[1] <= y[0] for x, y in zip(left, right)) and any(
        x[1] < y[0] for x, y in zip(left, right))


def zero_event_poisson_upper_bound(exposure_hours, confidence=.95):
    """One-sided zero-event HPP rate bound, in events/hour: -log(alpha)/T.

    Assumes a precommitted exposure and homogeneous Poisson process; it is not
    the per-session binomial bound. Nonrepresentable strictly positive final
    rates raise ValueError rather than become zero/infinity.
    """
    exposure = _comparison_value(exposure_hours)
    confidence = _comparison_value(confidence)
    if exposure <= 0 or not 0 < confidence < 1:
        raise ValueError('positive exposure and confidence strictly between 0 and 1 required')
    numerator = -math.log1p(-confidence)
    try:
        rate = numerator / exposure
    except OverflowError:
        rate = float(Fraction(numerator) / Fraction(exposure))
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError('positive Poisson rate exceeds float64 support')
    return rate


def tiny_time_constrained_edit_distance(reference_tokens, hypothesis_tokens,
                                       reference_intervals, hypothesis_intervals, collar=0):
    """Single-stream token DP with strict positive time overlap for diagonals.

    Each explicit word interval must have positive duration. Expand each
    reference interval by a nonnegative collar (seconds) on both sides;
    touching endpoints do not overlap. Insertions/deletions cost one and are
    unrestricted. This bounded model has no speaker permutation, timestamps
    inferred from utterance text, or production tcpWER wrapper.
    """
    reference = _tokens(reference_tokens, 'reference')
    hypothesis = _tokens(hypothesis_tokens, 'hypothesis')
    if max(len(reference), len(hypothesis)) > 24:
        raise ValueError('at most 24 tokens per side')
    ri = [_bounds(v) for v in _sequence(reference_intervals, 'reference intervals')]
    hi = [_bounds(v) for v in _sequence(hypothesis_intervals, 'hypothesis intervals')]
    if len(ri) != len(reference) or len(hi) != len(hypothesis):
        raise ValueError('one word interval per token required')
    if any(lo >= up for lo, up in ri + hi):
        raise ValueError('word intervals need positive duration')
    collar = _comparison_value(collar)
    if collar < 0:
        raise ValueError('collar must be nonnegative')
    # Exact rational arithmetic keeps large time origins and small collars from
    # silently changing a touching endpoint into an overlap (or vice versa).
    ri = [(Fraction(lo) - Fraction(collar), Fraction(up) + Fraction(collar)) for lo, up in ri]
    hi = [(Fraction(lo), Fraction(up)) for lo, up in hi]
    row = list(range(len(hypothesis) + 1))
    for i, (token, interval) in enumerate(zip(reference, ri), 1):
        old, row = row, [i]
        for j, (other, times) in enumerate(zip(hypothesis, hi), 1):
            overlap = max(interval[0], times[0]) < min(interval[1], times[1])
            diagonal = old[j-1] + (token != other) if overlap else math.inf
            row.append(min(row[-1] + 1, old[j] + 1, diagonal))
    return row[-1]
