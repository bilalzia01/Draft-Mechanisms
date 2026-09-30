"""Record-geometry ordinal bound and profilewise donor upper bound."""
from __future__ import annotations
import math
from functools import lru_cache
import numpy as np
from scipy.special import expit

def summary(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        raise ValueError('values must be nonempty')
    if np.all(values == values[0]):
        return (float(values[0]), 0.0)
    return (float(np.mean(values)), float(np.std(values, ddof=1) / math.sqrt(values.size)))

def water_filling_values_from_sorted(ordered: np.ndarray, cumulative: np.ndarray, targets: np.ndarray, kappa: float) -> np.ndarray:
    water_level = _water_levels(ordered, cumulative, kappa)
    return np.maximum(water_level - kappa * targets, 0.0)

def _water_levels(ordered: np.ndarray, cumulative: np.ndarray, kappa: float) -> np.ndarray:
    water_level = np.full(ordered.shape[0], np.nan, dtype=float)
    n = ordered.shape[1]
    for active in range(1, n + 1):
        candidate = (1.0 + kappa * cumulative[:, active - 1]) / active
        valid = candidate >= kappa * ordered[:, active - 1] - 1e-13
        if active < n:
            valid &= candidate <= kappa * ordered[:, active] + 1e-13
        valid &= np.isnan(water_level)
        water_level[valid] = candidate[valid]
    if np.isnan(water_level).any():
        raise AssertionError('water level not found')
    return water_level

class ExactGapOracle:

    def __init__(self, n: int, k: int, cache_size: int=16384):
        self.n, self.k = (n, k)
        self.maximum = k * (n - 1)
        self.total = k * n * (n - 1) // 2
        self.allowed = {}
        for teams in range(1, n + 1):
            remaining = n - teams
            low = max(k * teams * (teams - 1) // 2, self.total - remaining * self.maximum, 0)
            self.allowed[teams] = tuple(((1 << high - low + 1) - 1 << low if high >= low else 0 for score in range(self.maximum + 1) for high in [min(self.total, self.total - remaining * score)]))
        self.prefix = lru_cache(maxsize=cache_size)(self._prefix)

    def _prefix(self, composition: tuple[int, ...]) -> tuple[int, ...]:
        teams = sum(composition)
        block = composition[-1]
        masks = self.allowed[teams]
        if len(composition) == 1:
            return tuple((1 << block * score & masks[score] for score in range(self.maximum + 1)))
        previous = self.prefix(composition[:-1])
        reached = 0
        result = []
        for score in range(self.maximum + 1):
            result.append(reached << block * score & masks[score])
            reached |= previous[score]
        return tuple(result)

    def minimum_gaps(self, composition: tuple[int, ...]) -> tuple[int, ...]:
        if not composition or min(composition) < 1 or sum(composition) != self.n:
            raise ValueError('invalid composition')
        if len(composition) == 1:
            return ()
        weighted_rank = sum((i * block for i, block in enumerate(composition)))
        consecutive_base = (self.total - weighted_rank) / self.n
        answer, left_teams = ([], 0)
        for cut in range(1, len(composition)):
            left_teams += composition[cut - 1]
            left = self.prefix(composition[:cut])
            reverse = self.prefix(composition[cut:][::-1])
            shift = (self.n - left_teams) * self.maximum - self.total
            joined = tuple((bits >> shift if shift >= 0 else bits << -shift for bits in reverse))
            possible = sorted((score for score, bits in enumerate(left) if bits), key=lambda score: abs(score - (consecutive_base + cut - 1)))
            for distance in range(1, self.maximum + 1):
                if any((score + distance <= self.maximum and left[score] & joined[self.maximum - score - distance] for score in possible)):
                    answer.append(distance)
                    break
            else:
                raise ValueError('unrealizable composition')
        return tuple(answer)

def pattern_masks(records: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ordered = np.sort(records, axis=1)
    cuts = np.diff(ordered, axis=1) > 0
    weights = np.left_shift(np.uint32(1), np.arange(records.shape[1] - 1, dtype=np.uint32))
    masks = cuts.astype(np.uint32) @ weights
    unique, inverse = np.unique(masks, return_inverse=True)
    return (ordered, unique, inverse)

def virtual_scores_for_patterns(masks, n, k):
    oracle = ExactGapOracle(n, k)
    scores = np.empty((len(masks), n), dtype=np.int16)
    for index, mask in enumerate(masks):
        ends = [j + 1 for j in range(n - 1) if int(mask) & (1 << j)] + [n]
        sizes = tuple(np.diff([0] + ends).tolist())
        levels = np.r_[0, np.cumsum(oracle.minimum_gaps(sizes))]
        scores[index] = np.repeat(levels, sizes)
    return scores

def evaluate_universal_bound(records: np.ndarray, base_kappas: list[float]) -> dict[str, object]:
    ordered, masks, inverse = pattern_masks(records)
    virtual_by_pattern = virtual_scores_for_patterns(masks, records.shape[1], int(records.sum(axis=1)[0] * 2 // (records.shape[1] * (records.shape[1] - 1))))
    virtual = virtual_by_pattern[inverse]
    cumulative = np.cumsum(virtual, axis=1)
    target_position = np.sum(records < records[:, 0, None], axis=1)
    target = virtual[np.arange(len(records)), target_position]
    denominators = virtual_by_pattern[:, 1:] * np.arange(1, records.shape[1]) - np.cumsum(virtual_by_pattern, axis=1)[:, :-1]
    knots = {1 / int(value) for value in np.unique(denominators) if value > 0}
    budgets = sorted(knots | set(base_kappas))
    curve = []
    for budget in budgets:
        mean, se = summary(water_filling_values_from_sorted(virtual, cumulative, target, budget))
        curve.append({'kappa': budget, 'ordinal': mean, 'ordinal_se': se})
    dense = np.cumsum(np.column_stack([np.zeros(len(masks), dtype=int), np.diff(virtual_by_pattern, axis=1) > 0]), axis=1)
    return {'curve': curve, 'patterns': len(masks), 'nonunit_patterns': int(np.sum(np.any(virtual_by_pattern != dense, axis=1)))}

_BATCH_SIZE = 256

def _poisson_binomial_pmfs(donor_logs: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    batch, donor_count = donor_logs.shape
    maximum_wins = k * donor_count
    probabilities = expit(donor_logs[:, :, None] - donor_logs[:, None, :])
    pmfs = np.zeros((batch, donor_count, maximum_wins + 1), dtype=float)
    pmfs[:, :, 0] = 1.0
    degree = 0
    binomial_coefficients = np.asarray([math.comb(k, wins) for wins in range(k + 1)], dtype=float)
    for opponent in range(donor_count):
        probability = probabilities[:, :, opponent]
        kernel = np.stack([binomial_coefficients[wins] * probability ** wins * (1.0 - probability) ** (k - wins) for wins in range(k + 1)], axis=2)
        updated = np.zeros_like(pmfs)
        active = pmfs[:, :, :degree + 1]
        for wins in range(k + 1):
            updated[:, :, wins:wins + degree + 1] += active * kernel[:, :, wins, None]
        pmfs = updated
        degree += k
    return (pmfs, probabilities)

def _batch_donor_values(sorted_log_strengths: np.ndarray, k: int, kappas: np.ndarray) -> np.ndarray:
    batch, team_count = sorted_log_strengths.shape
    donor_count = team_count - 1
    maximum_wins = k * donor_count
    donor_logs = sorted_log_strengths[:, 1:]
    pmfs, donor_probabilities = _poisson_binomial_pmfs(donor_logs, k)
    weakest_mean = k * np.sum(expit(sorted_log_strengths[:, :1] - donor_logs), axis=1)
    costs = k * np.sum(donor_probabilities, axis=2) - weakest_mean[:, None]
    if np.any(np.diff(costs, axis=1) < -2e-12):
        raise AssertionError('donor costs must be nondecreasing')
    output = np.ones((batch, kappas.size), dtype=float)
    wins = np.arange(maximum_wins + 1, dtype=float)
    for low in range(donor_count):
        envelope = np.zeros((batch, maximum_wins + 1), dtype=float)
        cost_sum = np.zeros(batch, dtype=float)
        first_cost = costs[:, low]
        for high in range(low, donor_count):
            np.maximum(envelope, pmfs[:, high], out=envelope)
            cost_sum += costs[:, high]
            mass = np.sum(envelope, axis=1)
            first_moment = envelope @ wins
            count = high - low + 1
            denominator = count + mass
            intercept = mass / denominator
            slope = (cost_sum + first_moment - mass * weakest_mean) / denominator
            np.minimum(output, intercept[:, None] + slope[:, None] * kappas, out=output)
            cumulative = first_cost.copy()
            demand_count = 1
            np.minimum(output, (mass[:, None] + cumulative[:, None] * kappas) / demand_count, out=output)
            cumulative += first_cost
            demand_count += 1
            np.minimum(output, (mass[:, None] + cumulative[:, None] * kappas) / demand_count, out=output)
            for donor in range(low + 1, high + 1):
                cumulative += costs[:, donor]
                demand_count += 1
                np.minimum(output, (mass[:, None] + cumulative[:, None] * kappas) / demand_count, out=output)
    return np.minimum(output, 1.0)

def population_donor_values(sorted_log_strengths: np.ndarray, k: int, kappas: np.ndarray) -> np.ndarray:
    logs = np.asarray(sorted_log_strengths, dtype=float)
    grid = np.asarray(kappas, dtype=float)
    if logs.ndim != 2 or logs.shape[0] == 0 or logs.shape[1] < 2:
        raise ValueError('sorted_log_strengths must be a nonempty 2-D array')
    if not np.all(np.isfinite(logs)):
        raise ValueError('sorted_log_strengths must be finite')
    if np.any(np.diff(logs, axis=1) < 0.0):
        raise ValueError('log-strength profiles must be sorted nondecreasing')
    if isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)):
        raise TypeError('k must be a positive integer')
    if int(k) < 1:
        raise ValueError('k must be a positive integer')
    if grid.ndim != 1 or grid.size == 0:
        raise ValueError('kappas must be a nonempty one-dimensional array')
    if not np.all(np.isfinite(grid)) or np.any((grid < 0.0) | (grid > 1.0)):
        raise ValueError('kappas must be finite and lie in [0, 1]')
    logs = logs - logs[:, :1]
    output = np.empty((logs.shape[0], grid.size), dtype=float)
    for start in range(0, logs.shape[0], _BATCH_SIZE):
        stop = min(start + _BATCH_SIZE, logs.shape[0])
        output[start:stop] = _batch_donor_values(logs[start:stop], int(k), grid)
    return output
