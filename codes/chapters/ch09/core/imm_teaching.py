"""Scalar random-walk IMM teaching model; not a general motion tracker.

Modes share one local, unwrapped angular coordinate in degrees. Each call is
one fixed interval: Q and the row-stochastic transition already belong to it.
Keep mode mass in log space, including positive mass whose display underflows.
"""
from __future__ import annotations

import math
import numpy as np
from codes.chapters.ch02.core.conventions import finite_real_array, finite_real_scalar


def _logsum(values):
    high = float(np.max(values))
    if high == -math.inf:
        return high
    return high + math.log(math.fsum(math.exp(float(x)-high) for x in values))


class ScalarRandomWalkIMM:
    """Interaction, scalar KF update, mode evidence and moment fusion.

    Unreachable destination modes are rejected; no division by zero or silent
    invented state. None means no observation, so no likelihood is evaluated.
    This model has no angular wrap, birth/death, learned bias or audio input.
    """
    def __init__(self, probabilities, means, variances, transition,
                 process_variances, measurement_variance):
        mu = finite_real_array(probabilities, 'probabilities')
        self.means = finite_real_array(means, 'means').copy()
        self.variances = finite_real_array(variances, 'variances').copy()
        self.transition = finite_real_array(transition, 'transition').copy()
        self.process_variances = finite_real_array(process_variances, 'process_variances').copy()
        self.measurement_variance = finite_real_scalar(measurement_variance, 'measurement_variance')
        if (mu.ndim != 1 or not mu.size or np.any(mu < 0)
                or not math.isclose(math.fsum(mu), 1., rel_tol=1e-12, abs_tol=1e-12)):
            raise ValueError('probabilities must be a probability vector')
        n = mu.size
        if any(x.shape != (n,) for x in (self.means, self.variances, self.process_variances)):
            raise ValueError('each mode needs one mean, variance and process variance')
        if np.any(self.variances < 0) or np.any(self.process_variances < 0) or self.measurement_variance <= 0:
            raise ValueError('variances must be nonnegative and R strictly positive')
        if (self.transition.shape != (n, n) or np.any(self.transition < 0)
                or not np.allclose(self.transition.sum(axis=1), 1., rtol=1e-12, atol=1e-12)):
            raise ValueError('transition must be row-stochastic')
        self._log_mu = np.array([math.log(float(x)) if x else -math.inf for x in mu])
        self._log_mu -= _logsum(self._log_mu)

    @property
    def log_probabilities(self):
        return self._log_mu.copy()

    @property
    def probabilities(self):
        return np.array([math.exp(float(x)) for x in self._log_mu])

    def step(self, observation=None):
        """Advance one interval atomically; return all intermediate moments."""
        z = None if observation is None else finite_real_scalar(observation, 'observation')
        n = self.means.size
        log_t = np.array([[math.log(float(x)) if x else -math.inf for x in row]
                          for row in self.transition])
        joint = self._log_mu[:, None] + log_t
        log_c = np.array([_logsum(joint[:, j]) for j in range(n)])
        if np.any(~np.isfinite(log_c)):
            raise ValueError('each destination mode must be reachable')
        mixing = np.array([[math.exp(float(joint[i, j]-log_c[j])) for j in range(n)] for i in range(n)])
        with np.errstate(over='raise', invalid='raise', divide='raise'):
            try:
                mixed_mean = mixing.T @ self.means
                mixed_var = np.array([math.fsum(float(mixing[i, j]) *
                    (float(self.variances[i]) + float(self.means[i]-mixed_mean[j])**2)
                    for i in range(n)) for j in range(n)])
                prior_var = mixed_var + self.process_variances
                new_mean, new_var = mixed_mean.copy(), prior_var.copy()
                innovation = innovation_var = gain = log_likelihood = None
                new_log_mu = log_c.copy()
                if z is not None:
                    innovation = z-mixed_mean
                    innovation_var = prior_var+self.measurement_variance
                    gain = prior_var/innovation_var
                    log_likelihood = np.array([-.5*(math.log(2*math.pi)+math.log(float(s))+
                        (float(v)/math.sqrt(float(s)))**2) for v, s in zip(innovation, innovation_var)])
                    new_log_mu += log_likelihood
                    new_mean += gain*innovation
                    new_var = (1-gain)**2*prior_var+gain**2*self.measurement_variance
                if not all(np.all(np.isfinite(x)) for x in (new_mean, new_var, new_log_mu)):
                    raise ValueError('IMM moments or log evidence exceed float64 support')
                new_log_mu -= _logsum(new_log_mu)
                weights = np.array([math.exp(float(x)) for x in new_log_mu])
                overall_mean = math.fsum(float(w*m) for w, m in zip(weights, new_mean))
                overall_var = math.fsum(float(w)*(float(v)+float(m-overall_mean)**2)
                                       for w, m, v in zip(weights, new_mean, new_var))
                if not math.isfinite(overall_mean) or not math.isfinite(overall_var):
                    raise ValueError('fused moments exceed float64 support')
            except (FloatingPointError, OverflowError) as error:
                raise ValueError('IMM arithmetic exceeds float64 support') from error
        self.means, self.variances, self._log_mu = new_mean, new_var, new_log_mu
        return {'observation': z, 'mode_prior': np.array([math.exp(v) for v in log_c]), 'log_mode_prior': log_c,
                'mixing': mixing, 'mixed_means': mixed_mean, 'mixed_variances': mixed_var,
                'predicted_means': mixed_mean.copy(), 'predicted_variances': prior_var,
                'innovation': innovation, 'innovation_variances': innovation_var, 'gain': gain,
                'log_likelihood': log_likelihood, 'mode_probabilities': weights,
                'log_mode_probabilities': new_log_mu.copy(),
                'display_underflow_indices': np.flatnonzero((weights == 0)&np.isfinite(new_log_mu)).tolist(),
                'means': new_mean.copy(), 'variances': new_var.copy(),
                'fused_mean': overall_mean, 'fused_variance': overall_var}


def imm_missing_observation_example():
    model = ScalarRandomWalkIMM([.8, .2], [0., 0.], [1., 1.],
        [[.9, .1], [.2, .8]], [0., 4.], 1.)
    return {'interval_s': 1., 'unit': 'local degrees and deg^2',
            'observations': [3., None, None],
            'steps': [model.step(z) for z in (3., None, None)],
            'scope': 'known scalar random-walk modes; no bias estimation, audio or calibrated maneuver probability'}
