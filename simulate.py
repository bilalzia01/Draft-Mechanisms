"""Refit strengths and reproduce both accuracy experiments."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from fractions import Fraction
from functools import lru_cache
from pathlib import Path
import numpy as np
from scipy.special import expit
from bounds import water_filling_values_from_sorted, evaluate_universal_bound, population_donor_values
from calibrate import pooled_calibration

F = Fraction

def schedule_template() -> np.ndarray:
    schedule = np.zeros((30, 30), dtype=np.int16)
    for i in range(30):
        conference_i, within_i = divmod(i, 15)
        division_i, slot_i = divmod(within_i, 5)
        for j in range(i + 1, 30):
            conference_j, within_j = divmod(j, 15)
            division_j, slot_j = divmod(within_j, 5)
            if conference_i != conference_j:
                meetings = 2
            elif division_i == division_j:
                meetings = 4
            else:
                difference = (slot_j - slot_i) % 5 if division_i < division_j else (slot_i - slot_j) % 5
                meetings = 4 if difference in (0, 1, 2) else 3
            schedule[i, j] = schedule[j, i] = meetings
    return schedule

def rank_coefficients(records: np.ndarray, targets: np.ndarray) -> np.ndarray:
    target_records = records[np.arange(len(records)), targets][:, None]
    higher = np.sum(records > target_records, axis=1)
    tied = np.sum(records == target_records, axis=1)
    ranks = np.arange(records.shape[1])[None, :]
    occupied = (ranks >= higher[:, None]) & (ranks < (higher + tied)[:, None])
    return occupied / tied[:, None]

@lru_cache(maxsize=None)
def _rank_flip_rows(n: int, maximum_record: int, total_wins: int) -> np.ndarray:
    specifications = set()
    for own in range(maximum_record):
        categories = []
        if own >= 2:
            categories.append((1, own - 1, 0, 0, 0, 0))
        if own >= 1:
            categories.append((own, own, 0, 1, 0, 0))
        if own + 1 <= maximum_record:
            categories.append((own + 1, own + 1, 1, 0, 0, 0))
        if own + 2 <= maximum_record:
            categories.append((own + 2, own + 2, 1, 0, 0, 1))
        if own + 3 <= maximum_record:
            categories.append((own + 3, maximum_record, 1, 0, 1, 0))
        for higher in range(n - 1):
            for one_above in range(n - 1 - higher):
                for tied in range(n - 1 - higher - one_above):
                    lower = n - 2 - higher - one_above - tied
                    if lower and own == 0:
                        continue
                    fixed = own + one_above * (own + 1) + tied * own
                    minimum = fixed + higher * (own + 2)
                    maximum = fixed + higher * maximum_record + lower * (own - 1)
                    for low, high, above, opponent_tied, stays_higher, joins in categories:
                        if max(low, total_wins - maximum) > min(high, total_wins - minimum):
                            continue
                        specifications.add((higher + one_above + above, 1 + tied + opponent_tied, higher + stays_higher, 1 + one_above + joins))
    rows = []
    for before_start, before_size, after_start, after_size in specifications:
        row = np.zeros(n, dtype=float)
        row[before_start:before_start + before_size] += 1 / before_size
        row[after_start:after_start + after_size] -= 1 / after_size
        rows.append(row)
    result = np.unique(np.round(np.asarray(rows), 14), axis=0)
    result.flags.writeable = False
    return result

def rank_flip_rows(n: int=30, maximum_record: int=82, total_wins: int=1230) -> np.ndarray:
    return _rank_flip_rows(n, maximum_record, total_wins)

def current_rank_odds() -> tuple[Fraction, ...]:
    worst_to_best = tuple((F(str(value)) for value in (0.14, 0.14, 0.14, 0.125, 0.105, 0.09, 0.075, 0.06, 0.045, 0.03, 0.02, 0.015, 0.01, 0.005)))
    return (F(0),) * 16 + worst_to_best[::-1]

def upcoming_rank_odds() -> tuple[Fraction, ...]:
    return (F(0),) * 14 + (F(1, 37),) * 2 + (F(2, 37),) * 4 + (F(3, 37),) * 7 + (F(2, 37),) * 3

@lru_cache(maxsize=None)
def exact_ic_threshold(odds: tuple[Fraction, ...]) -> Fraction:
    maximum = F(0)
    for row in rank_flip_rows():
        value = sum((F(float(coefficient)).limit_denominator(900) * probability for coefficient, probability in zip(row, odds)), F(0))
        maximum = max(maximum, value)
    return maximum

def estimate(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=float)
    mean = float(np.mean(values))
    se = float(np.std(values, ddof=1) / math.sqrt(values.size))
    return {'accuracy': mean, 'se': se, 'ci95_low': max(0.0, mean - 1.96 * se), 'ci95_high': min(1.0, mean + 1.96 * se)}

def draw_log_strengths(draws, n, tau, seed):
    if draws < 2 or n < 2 or (not np.isfinite(tau)) or (tau <= 0):
        raise ValueError('positive spread and at least two teams/draws required')
    return np.random.default_rng(seed).normal(0, tau, (draws, n))

def sample_records(log_strengths, schedule, seed):
    logs, schedule = (np.asarray(log_strengths), np.asarray(schedule))
    if logs.ndim != 2 or schedule.shape != (logs.shape[1], logs.shape[1]) or (not np.all(np.isfinite(logs))) or np.any(schedule < 0) or np.any(schedule != schedule.astype(int)) or np.any(np.diag(schedule)) or np.any(schedule != schedule.T):
        raise ValueError('invalid strengths or schedule')
    rng = np.random.default_rng(seed)
    records = np.zeros(logs.shape, dtype=np.int16)
    for i in range(logs.shape[1]):
        for j in range(i + 1, logs.shape[1]):
            meetings = int(schedule[i, j])
            wins = rng.binomial(meetings, expit(logs[:, i] - logs[:, j])).astype(np.int16)
            records[:, i] += wins
            records[:, j] += meetings - wins
    return records

def mmu_values(records, weakest, kappa):
    ordered = np.sort(records, axis=1).astype(float)
    level = np.min((1 + kappa * np.cumsum(ordered, axis=1)) / np.arange(1, records.shape[1] + 1), axis=1)
    return np.maximum(level - kappa * records[np.arange(len(records)), weakest], 0)

def free_values(records, weakest):
    minimum = records.min(axis=1)
    return np.where(records[np.arange(len(records)), weakest] == minimum, 1 / np.sum(records == minimum[:, None], axis=1), 0)

def rank_rule_values(records, weakest, odds):
    return rank_coefficients(records, weakest) @ np.asarray(odds, dtype=float)

def gain_estimate(values):
    result = estimate(values)
    result['ci95_low'] = result['accuracy'] - 1.96 * result['se']
    result['ci95_high'] = result['accuracy'] + 1.96 * result['se']
    return result

def curve(records, weakest, kappas):
    ordered = np.sort(records, axis=1).astype(float)
    cumulative = np.cumsum(ordered, axis=1)
    target = records[np.arange(len(records)), weakest]
    return [{'kappa': float(kappa), **estimate(water_filling_values_from_sorted(ordered, cumulative, target, kappa))} for kappa in kappas]

def save_json(path, data):
    path.write_text(json.dumps(data, indent=2) + '\n')

def run(output, draws=300000, h2h_draws=10000, seed=2026092903):
    output = Path(output)
    if output.exists():
        raise FileExistsError(f'refusing to overwrite {output}')
    if draws < 2 or h2h_draws < 2:
        raise ValueError('at least two draws required')
    output.mkdir(parents=True)
    calibration = pooled_calibration()
    identity = hashlib.sha256(json.dumps(calibration, sort_keys=True).encode()).hexdigest()
    common = {'calibration': calibration, 'calibration_sha256': identity, 'draws': draws, 'seed': seed, 'uncertainty': 'Monte Carlo SE over iid strength/outcome draws, conditional on fitted tau; not calibration uncertainty.', 'strengths_known_to_rules': False, 'smoothing': False}
    save_json(output / 'specification.json', common)
    logs = draw_log_strengths(draws, 30, calibration['tau'], seed)
    weakest = np.argmin(logs, axis=1)
    balanced_schedule = 3 * (np.ones((30, 30), dtype=int) - np.eye(30, dtype=int))
    balanced_records = sample_records(logs, balanced_schedule, seed + 1)
    nba_records = sample_records(logs, schedule_template(), seed + 2)
    print('pooled calibration and both schedules sampled', flush=True)
    kappas = kappa_grid()
    figure1 = {**common, 'n': 30, 'k': 3, 'games_per_team': 87, 'mmu': curve(balanced_records, weakest, kappas), 'unconstrained': estimate(free_values(balanced_records, weakest)), 'record_based_optimality': 'Balanced neutral Bradley-Terry with exchangeable iid prior'}
    save_json(output / 'figure-1.json', figure1)
    print('balanced MMU and unconstrained complete', flush=True)
    odds = {'current': current_rank_odds(), 'upcoming': upcoming_rank_odds()}
    figure2 = {**common, 'n': 30, 'games_per_team': 82, 'schedule': schedule_template().tolist(), 'scope': 'Static overall-rank NBA approximations with tied-slot averaging; not complete eligibility, Play-In or traded-pick rules. MMU optimality is not claimed for this unbalanced schedule.', 'mmu': curve(nba_records, weakest, kappas), 'rules': {}}
    for name, vector in odds.items():
        threshold = exact_ic_threshold(vector)
        values = rank_rule_values(nba_records, weakest, vector)
        paired_gain = mmu_values(nba_records, weakest, float(threshold)) - values
        figure2['rules'][name] = {'estimate': estimate(values), 'odds_best_to_worst': [str(p) for p in vector], 'ic_threshold': str(threshold), 'gain_at_ic_threshold': gain_estimate(paired_gain), 'gain_scope': 'Paired Monte Carlo estimate at the threshold, not a guaranteed minimum.'}
    save_json(output / 'figure-2.json', figure2)
    print('NBA comparison complete', flush=True)
    reordered = np.take_along_axis(balanced_records, np.argsort(logs, axis=1), axis=1)
    print('evaluating ordinal geometry', flush=True)
    figure1['ordinal'] = evaluate_universal_bound(reordered, kappas)
    save_json(output / 'figure-1.json', figure1)
    print('ordinal bound complete', flush=True)
    donor_logs = np.sort(draw_log_strengths(h2h_draws, 30, calibration['tau'], seed + 3), axis=1)
    values = population_donor_values(donor_logs, 3, np.asarray(kappas))
    free = figure1['unconstrained']
    figure1['full_information'] = {'draws': h2h_draws, 'seed': seed + 3, 'formula': 'min(E_S[profilewise donor upper bound], population unconstrained Bayes accuracy)', 'numerics': 'Donor bounds evaluated in floating point; population expectation estimated by Monte Carlo.', 'curve': []}
    for j, kappa in enumerate(kappas):
        component = estimate(values[:, j])
        figure1['full_information']['curve'].append({
            'kappa': kappa,
            'analytic_expectation': component,
            'capped_estimate': min(component['accuracy'], free['accuracy']),
            'approximate_95_component_interval': [
                min(component['accuracy'] - 2.2414 * component['se'],
                    free['accuracy'] - 2.2414 * free['se']),
                min(component['accuracy'] + 2.2414 * component['se'],
                    free['accuracy'] + 2.2414 * free['se']),
            ],
        })
    figure1['complete'] = figure2['complete'] = True
    save_json(output / 'figure-1.json', figure1)
    save_json(output / 'figure-2.json', figure2)
    print('population donor bound complete', flush=True)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--draws', type=int, default=300000)
    parser.add_argument('--h2h-draws', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=2026092903)
    args = parser.parse_args()
    run(args.output, args.draws, args.h2h_draws, args.seed)

def kappa_grid():
    base = [0.0, 1e-05, 2e-05, 5e-05, 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16, 0.17, 0.18, 0.19, 0.2, 0.21, 0.22, 0.23, 0.24, 0.25, 0.26, 0.27, 0.28, 0.29, 0.3, 0.31, 0.32, 0.33, 0.3333333333333333, 0.34, 0.35000000000000003, 0.36, 0.37, 0.38, 0.39, 0.4, 0.41000000000000003, 0.42, 0.43, 0.44, 0.45, 0.46, 0.47000000000000003, 0.48, 0.49, 0.5, 0.51, 0.52, 0.53, 0.54, 0.55, 0.56, 0.5700000000000001, 0.58, 0.59, 0.6, 0.61, 0.62, 0.63, 0.64, 0.65, 0.66, 0.67, 0.68, 0.6900000000000001, 0.7000000000000001, 0.71, 0.72, 0.73, 0.74, 0.75, 0.76, 0.77, 0.78, 0.79, 0.8, 0.81, 0.8200000000000001, 0.8300000000000001, 0.84, 0.85, 0.86, 0.87, 0.88, 0.89, 0.9, 0.91, 0.92, 0.93, 0.9400000000000001, 0.9500000000000001, 0.96, 0.97, 0.98, 0.99, 1.0]
    return sorted(set(base) | set(np.linspace(0, .2, 101)) | {16/135, 221/3256})

if __name__ == '__main__':
    main()
