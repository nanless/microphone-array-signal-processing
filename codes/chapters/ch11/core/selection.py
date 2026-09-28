"""Small deterministic selection helpers; no device performance is inferred.

Intervals are assumed valid bounds, not automatically confidence intervals.
Token scoring is exhaustive teaching enumeration (at most 4 streams, 6 reference
utterances, 24 tokens per side), not a replacement for a production WER scorer.
"""
from itertools import permutations, product
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


def upper_limit_verdict(intervals, limits):
    """All metrics are minimized; None means unmeasured, equality is allowed."""
    if len(intervals) != len(limits) or len(limits) == 0:
        raise ValueError('provide equally sized nonempty intervals and limits')
    statuses = []
    for interval, limit in zip(intervals, limits):
        limit = _comparison_value(limit)
        if interval is None:
            statuses.append('undetermined')
            continue
        if len(interval) != 2:
            raise ValueError('an interval needs lower and upper bounds')
        lower, upper = [_comparison_value(v) for v in interval]
        if lower > upper:
            raise ValueError('lower must not exceed upper')
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
    row = list(range(len(hypothesis) + 1))
    for i, token in enumerate(reference, 1):
        previous, row = row, [i]
        for j, other in enumerate(hypothesis, 1):
            row.append(min(row[-1] + 1, previous[j] + 1,
                           previous[j-1] + (token != other)))
    return row[-1]


def small_slot_word_errors(utterances, speakers, hypotheses):
    """Compare global speaker permutation with utterance-to-slot assignment.

    utterances are ordered token lists, with matching integer speaker labels.
    No timestamps or time-constrained matching; reference order is supplied by
    the caller. ORC concatenates assigned utterances in that order. cp pads
    empty streams to the larger speaker/slot count. Missing words are deletions.
    """
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
