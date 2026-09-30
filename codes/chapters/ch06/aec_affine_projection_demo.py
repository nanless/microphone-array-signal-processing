"""E06-34--38: causal APA windows, regularization, freeze and path priors.

Run python -m codes.chapters.ch06.aec_affine_projection_demo. Inputs are
dimensionless deterministic arithmetic, not speech or device measurements.
"""

from __future__ import annotations

if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[3]))

import json

import numpy as np

from codes.chapters.ch06.core.aec_affine_projection import APAState


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def _two_columns(delta=1., mu=1., initial_weights=None):
    state = APAState(2, 2, step_size=mu, regularization=delta,
                     initial_weights=initial_weights)
    first = state.step(1., .5, freeze=True)
    second = state.step(2., 1.25)
    return state, first, second


def projection_hand_case() -> dict:
    """E34: warm the first TRUE observation while holding the zero weights."""
    _, first, result = _two_columns()
    _, _, exact = _two_columns(delta=0.)
    one = APAState(2, 1, step_size=1., regularization=1., initial_history=[1.])
    one_result = one.step(2., 1.25)
    return _plain({
        "initial_observation_frozen": first, "regularized": result,
        "unregularized_full_rank": exact, "single_column": one_result,
        "exact_hand_values": {
            "gram_plus_identity": [[6, 2], [2, 2]],
            "inverse": [["1/4", "-1/4"], ["-1/4", "3/4"]],
            "small_system_solution": ["3/16", "1/16"],
            "increment": ["7/16", "3/16"],
            "posterior_errors": ["3/16", "1/16"],
        },
    })


def same_weight_errors() -> dict:
    """E35: an old emitted prior error is not a current-window prior error."""
    state = APAState(2, 2, step_size=1., regularization=1.)
    first = state.step(1., .5)
    second = state.step(2., 1.25)
    # Independent two-by-two hand inverse. The wrong input reuses e0=1/2;
    # the correct old observation error at the NEW prior weights is 1/4.
    wrong_error = np.array([.75, first["prior_error"]])
    wrong_alpha = np.array([.25 * wrong_error[0] - .25 * wrong_error[1],
                           -.25 * wrong_error[0] + .75 * wrong_error[1]])
    wrong_increment = np.array([2. * wrong_alpha[0] + wrong_alpha[1], wrong_alpha[0]])
    return _plain({
        "first": first, "second": second,
        "incorrect_reused_error_window": wrong_error,
        "incorrect_increment": wrong_increment,
        "incorrect_posterior_weights": first["posterior_weights"] + wrong_increment,
    })


def freeze_window_control() -> dict:
    """E36: a frozen contaminated observation remains until explicitly cleared."""
    polluted = APAState(2, 2, step_size=1., regularization=1.)
    held = polluted.step(1., 10., freeze=True)
    resumed = polluted.step(2., 1.25)
    cleared = APAState(2, 2, step_size=1., regularization=1.)
    cleared.step(1., 10., freeze=True)
    cleared.reset_projection_history()
    history_after_clear = cleared.history
    clean = cleared.step(2., 1.25)
    return _plain({
        "frozen_contaminated": held, "resumed_with_old_observation": resumed,
        "render_history_after_projection_clear": history_after_clear,
        "resumed_after_projection_clear": clean,
        "K1_relation": "U has one column: delta equals the NLMS denominator floor",
    })


def rank_and_units() -> dict:
    """E37: regularization permits a solve but contributes no new direction."""
    singular = APAState(2, 2, step_size=1., regularization=0., initial_history=[1.])
    singular.step(1., 1., freeze=True)
    try:
        singular.step(1., 1.)
    except ValueError as error:
        rejected = str(error)
    else:
        raise AssertionError("the deficient zero-regularization window was accepted")
    cases = {}
    for name, scale, delta in (("original", 1., 1.),
                               ("scaled_with_floor", 10., 100.),
                               ("scaled_without_floor", 10., 1.)):
        state = APAState(2, 2, step_size=1., regularization=delta,
                         initial_history=[scale])
        state.step(scale, scale, freeze=True)
        cases[name] = state.step(scale, scale)
    return _plain({"unregularized_rejection": rejected, "cases": cases,
                   "input_scale": 10., "matching_regularization_scale": 100.})


def path_prior_and_holdout() -> dict:
    """E38: correct/wrong external priors choose solutions; neither raises rank."""
    truth = np.array([1., 2.])
    heldout_u = np.array([1., -1.])
    cases = {}
    for name, prior in (("correct_prior", [1., 2.]),
                        ("wrong_prior", [2., 1.]), ("zero_prior", [0., 0.])):
        state = APAState(2, 2, step_size=1., regularization=1.,
                         initial_history=[1.], initial_weights=prior)
        state.step(1., 3., freeze=True)
        trained = state.step(1., 3.)
        estimate = float(heldout_u @ state.weights)
        cases[name] = {"training": trained,
                       "heldout_prior_echo": estimate,
                       "heldout_prior_error": float(heldout_u @ truth) - estimate}
    return _plain({
        "truth": truth, "training_projection_matrix": [[1., 1.], [1., 1.]],
        "training_rank": 1, "null_direction": [1., -1.],
        "heldout_regressor": heldout_u, "heldout_echo_truth": -1.,
        "holdout_adaptation": "disabled; no gain/delay fit", "cases": cases,
    })


def run_demo() -> dict:
    return {
        "scope": "dimensionless causal APA arithmetic, not industrial AEC performance",
        "E06-34": projection_hand_case(), "E06-35": same_weight_errors(),
        "E06-36": freeze_window_control(), "E06-37": rank_and_units(),
        "E06-38": path_prior_and_holdout(),
    }


if __name__ == "__main__":
    print(json.dumps(run_demo(), ensure_ascii=False, indent=2, allow_nan=False))
