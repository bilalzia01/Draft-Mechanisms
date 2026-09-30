"""Pooled early-season Bradley-Terry fit by Laplace marginal likelihood."""
from __future__ import annotations
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Sequence
import numpy as np
from scipy import optimize, special

ROOT = Path(__file__).resolve().parent

DEFAULT_TAU_BOUNDS = (0.0001, 3.0)

DEFAULT_HOME_ADVANTAGE_BOUNDS = (-2.5, 2.5)

@dataclass(frozen=True)
class SeasonGames:
    season: str
    teams: tuple[str, ...]
    home: np.ndarray
    away: np.ndarray
    outcome: np.ndarray
    neutral: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.season, str) or not self.season.strip():
            raise ValueError('season must be a nonempty string')
        teams = tuple(self.teams)
        if len(teams) < 2:
            raise ValueError('a season needs at least two teams')
        if any((not isinstance(team, str) or not team for team in teams)):
            raise ValueError('team names must be nonempty strings')
        if len(set(teams)) != len(teams):
            raise ValueError('team names must be unique within a season')
        home_raw = np.asarray(self.home)
        away_raw = np.asarray(self.away)
        outcome_raw = np.asarray(self.outcome)
        for name, array in (('home', home_raw), ('away', away_raw), ('outcome', outcome_raw)):
            if array.ndim != 1:
                raise ValueError(f'{name} must be one-dimensional')
        if not home_raw.size == away_raw.size == outcome_raw.size:
            raise ValueError('home, away, and outcome must have the same length')
        if home_raw.size == 0:
            raise ValueError('a season needs at least one game')
        if not np.issubdtype(home_raw.dtype, np.integer) or not np.issubdtype(away_raw.dtype, np.integer):
            raise ValueError('home and away must be integer arrays')
        home = np.array(home_raw, dtype=np.int64, copy=True)
        away = np.array(away_raw, dtype=np.int64, copy=True)
        if np.any(home < 0) or np.any(home >= len(teams)):
            raise ValueError('home contains an out-of-range team index')
        if np.any(away < 0) or np.any(away >= len(teams)):
            raise ValueError('away contains an out-of-range team index')
        if np.any(home == away):
            raise ValueError('a team cannot play itself')
        if not (np.issubdtype(outcome_raw.dtype, np.bool_) or np.issubdtype(outcome_raw.dtype, np.integer) or np.issubdtype(outcome_raw.dtype, np.floating)):
            raise ValueError('outcome must be a binary numeric array')
        if not np.all(np.isfinite(outcome_raw)) or not np.all((outcome_raw == 0) | (outcome_raw == 1)):
            raise ValueError('outcome values must be zero or one')
        outcome = np.array(outcome_raw, dtype=float, copy=True)
        if self.neutral is None:
            neutral = np.zeros(home.size, dtype=bool)
        else:
            neutral_raw = np.asarray(self.neutral)
            if neutral_raw.ndim != 1 or neutral_raw.size != home.size:
                raise ValueError('neutral must be one-dimensional and match the games')
            if not (np.issubdtype(neutral_raw.dtype, np.bool_) or (np.issubdtype(neutral_raw.dtype, np.integer) and np.all((neutral_raw == 0) | (neutral_raw == 1)))):
                raise ValueError('neutral must be a boolean array')
            neutral = np.array(neutral_raw, dtype=bool, copy=True)
        for array in (home, away, outcome, neutral):
            array.flags.writeable = False
        object.__setattr__(self, 'teams', teams)
        object.__setattr__(self, 'home', home)
        object.__setattr__(self, 'away', away)
        object.__setattr__(self, 'outcome', outcome)
        object.__setattr__(self, 'neutral', neutral)

    @property
    def n_teams(self) -> int:
        return len(self.teams)

    @property
    def n_games(self) -> int:
        return int(self.home.size)

@dataclass(frozen=True)
class FitDiagnostics:
    approximation: str
    converged: bool
    optimizer_message: str
    season_count: int
    game_count: int
    outer_iterations: int
    objective_evaluations: int
    log_marginal_likelihood: float
    maximum_mode_gradient: float
    tau_lower_bound: float
    tau_upper_bound: float
    tau_at_boundary: bool
    home_advantage_at_boundary: bool

@dataclass(frozen=True)
class PopulationFit:
    home_advantage: float
    tau: float
    diagnostics: FitDiagnostics

@dataclass(frozen=True)
class _SeasonDesign:
    season: str
    teams: tuple[str, ...]
    contrast: np.ndarray
    home_indicator: np.ndarray
    outcome: np.ndarray
    basis: np.ndarray

@dataclass(frozen=True)
class _ModeResult:
    coordinates: np.ndarray
    value: float
    hessian: np.ndarray
    iterations: int
    maximum_gradient: float

@lru_cache(maxsize=None)
def _zero_sum_basis(n: int) -> np.ndarray:
    basis = np.zeros((n, n - 1), dtype=float)
    for column in range(n - 1):
        denominator = math.sqrt((column + 1) * (column + 2))
        basis[:column + 1, column] = 1.0 / denominator
        basis[column + 1, column] = -(column + 1) / denominator
    basis.flags.writeable = False
    return basis

def _make_design(season: SeasonGames) -> _SeasonDesign:
    basis = _zero_sum_basis(season.n_teams)
    contrast = basis[season.home] - basis[season.away]
    home_indicator = (~season.neutral).astype(float)
    outcome = np.asarray(season.outcome, dtype=float)
    for array in (contrast, home_indicator, outcome):
        array.flags.writeable = False
    return _SeasonDesign(season=season.season, teams=season.teams, contrast=contrast, home_indicator=home_indicator, outcome=outcome, basis=basis)

def _conditional_terms(design: _SeasonDesign, coordinates: np.ndarray, *, home_advantage: float, tau: float) -> tuple[float, np.ndarray, np.ndarray]:
    x = np.asarray(coordinates, dtype=float)
    dimension = design.basis.shape[1]
    if x.shape != (dimension,):
        raise ValueError(f'coordinates must have shape ({dimension},)')
    if not math.isfinite(home_advantage):
        raise ValueError('home_advantage must be finite')
    if not math.isfinite(tau) or tau <= 0.0:
        raise ValueError('tau must be positive and finite')
    inverse_variance = 1.0 / (tau * tau)
    linear = design.home_indicator * home_advantage + design.contrast @ x
    probability = special.expit(linear)
    log_likelihood = float(np.sum(design.outcome * linear - np.logaddexp(0.0, linear)))
    value = log_likelihood - 0.5 * inverse_variance * float(x @ x)
    gradient = design.contrast.T @ (design.outcome - probability) - inverse_variance * x
    weights = probability * (1.0 - probability)
    hessian = design.contrast.T @ (weights[:, None] * design.contrast)
    hessian.flat[::dimension + 1] += inverse_variance
    return (value, gradient, hessian)

def _solve_mode(design: _SeasonDesign, home_advantage: float, tau: float, *, tolerance: float=1e-08, maximum_iterations: int=80) -> _ModeResult:
    dimension = design.basis.shape[1]
    coordinates = np.zeros(dimension, dtype=float)
    for iteration in range(1, maximum_iterations + 1):
        value, gradient, hessian = _conditional_terms(design, coordinates, home_advantage=home_advantage, tau=tau)
        maximum_gradient = float(np.max(np.abs(gradient)))
        if maximum_gradient <= tolerance:
            return _ModeResult(coordinates=coordinates, value=value, hessian=hessian, iterations=iteration - 1, maximum_gradient=maximum_gradient)
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError as error:
            raise RuntimeError(f'posterior Hessian failed for {design.season}') from error
        directional_gain = float(gradient @ step)
        if directional_gain <= 1e-12:
            candidate = coordinates + step
            candidate_value, candidate_gradient, candidate_hessian = _conditional_terms(design, candidate, home_advantage=home_advantage, tau=tau)
            roundoff = 32.0 * np.finfo(float).eps * (1.0 + abs(value))
            if candidate_value >= value - roundoff:
                return _ModeResult(coordinates=candidate, value=candidate_value, hessian=candidate_hessian, iterations=iteration, maximum_gradient=float(np.max(np.abs(candidate_gradient))))
            return _ModeResult(coordinates=coordinates, value=value, hessian=hessian, iterations=iteration - 1, maximum_gradient=maximum_gradient)
        scale = 1.0
        accepted = False
        while scale >= 2.0 ** (-24):
            candidate = coordinates + scale * step
            candidate_value = _conditional_terms(design, candidate, home_advantage=home_advantage, tau=tau)[0]
            if candidate_value >= value + 0.0001 * scale * directional_gain:
                coordinates = candidate
                accepted = True
                break
            scale *= 0.5
        if not accepted:
            raise RuntimeError(f'posterior mode line search stalled for {design.season}')
    maximum_gradient = float(np.max(np.abs(_conditional_terms(design, coordinates, home_advantage=home_advantage, tau=tau)[1])))
    raise RuntimeError(f'posterior mode failed to converge for {design.season}; maximum gradient {maximum_gradient:.3g}')

def _season_laplace(design: _SeasonDesign, home_advantage: float, tau: float) -> tuple[float, _ModeResult]:
    mode = _solve_mode(design, home_advantage, tau)
    sign, log_determinant = np.linalg.slogdet(mode.hessian)
    if sign <= 0.0 or not math.isfinite(float(log_determinant)):
        raise RuntimeError(f'non-positive posterior Hessian for {design.season}')
    dimension = design.basis.shape[1]
    log_marginal = mode.value - dimension * math.log(tau) - 0.5 * float(log_determinant)
    return (log_marginal, mode)

def fit_population(seasons: Sequence[SeasonGames], *, tau_bounds: tuple[float, float]=DEFAULT_TAU_BOUNDS, home_advantage_bounds: tuple[float, float]=DEFAULT_HOME_ADVANTAGE_BOUNDS) -> PopulationFit:
    records = tuple(seasons)
    if not records:
        raise ValueError('fit_population needs at least one season')
    if any((not isinstance(season, SeasonGames) for season in records)):
        raise TypeError('all seasons must be SeasonGames instances')
    names = [season.season for season in records]
    if len(set(names)) != len(names):
        raise ValueError('season identifiers must be unique')
    tau_lower, tau_upper = map(float, tau_bounds)
    home_lower, home_upper = map(float, home_advantage_bounds)
    if not (0.0 < tau_lower < tau_upper and math.isfinite(tau_upper)):
        raise ValueError('tau_bounds must be finite, positive, and increasing')
    if not (math.isfinite(home_lower) and math.isfinite(home_upper) and (home_lower < home_upper)):
        raise ValueError('home_advantage_bounds must be finite and increasing')
    designs = tuple((_make_design(season) for season in records))
    nonneutral_outcomes = np.concatenate([season.outcome[~season.neutral] for season in records])
    if nonneutral_outcomes.size == 0:
        raise ValueError('home advantage is unidentified when every game is neutral')
    if np.unique(nonneutral_outcomes).size < 2:
        raise ValueError('home advantage is separated without both home wins and losses')
    evaluations = 0
    cache: dict[tuple[float, float], float] = {}

    def objective(parameters: np.ndarray | Sequence[float]) -> float:
        nonlocal evaluations
        home_advantage = float(parameters[0])
        log_tau = float(parameters[1])
        key = (home_advantage, log_tau)
        cached = cache.get(key)
        if cached is not None:
            return cached
        tau = math.exp(log_tau)
        total = 0.0
        for design in designs:
            total += _season_laplace(design, home_advantage, tau)[0]
        result = -total
        if not math.isfinite(result):
            raise RuntimeError('Laplace objective became non-finite')
        cache[key] = result
        evaluations += 1
        return result
    home_rate = float(np.mean(nonneutral_outcomes))
    empirical_home = math.log(home_rate / (1.0 - home_rate))
    empirical_home = float(np.clip(empirical_home, home_lower + 0.05, home_upper - 0.05))
    log_tau_bounds = (math.log(tau_lower), math.log(tau_upper))
    bounds = ((home_lower, home_upper), log_tau_bounds)
    starts = (np.array([empirical_home, math.log(min(max(0.45, tau_lower), tau_upper))]), np.array([empirical_home, math.log(min(max(0.08, tau_lower), tau_upper))]))
    candidates: list[optimize.OptimizeResult] = []
    for start in starts:
        result = optimize.minimize(objective, start, method='L-BFGS-B', bounds=bounds, options={'ftol': 1e-12, 'gtol': 2e-07, 'maxiter': 200, 'maxls': 40})
        if result.success and np.all(np.isfinite(result.x)) and math.isfinite(float(result.fun)):
            candidates.append(result)
    boundary_home = optimize.minimize_scalar(lambda value: objective((value, log_tau_bounds[0])), bounds=(home_lower, home_upper), method='bounded', options={'xatol': 1e-10, 'maxiter': 200})
    if boundary_home.success and math.isfinite(float(boundary_home.fun)):
        candidates.append(optimize.OptimizeResult(x=np.array([boundary_home.x, log_tau_bounds[0]]), fun=boundary_home.fun, success=True, message='profile optimum at the guarded tau lower bound', nit=boundary_home.nfev))
    if not candidates:
        raise RuntimeError('population Laplace optimization failed to converge')
    best = min(candidates, key=lambda result: float(result.fun))
    home_advantage = float(best.x[0])
    tau = math.exp(float(best.x[1]))
    final_modes = [_season_laplace(design, home_advantage, tau)[1] for design in designs]
    maximum_mode_gradient = max((mode.maximum_gradient for mode in final_modes))
    tau_at_boundary = bool(tau <= tau_lower * (1.0 + 1e-06) or tau >= tau_upper * (1.0 - 1e-06))
    home_at_boundary = bool(home_advantage <= home_lower + 1e-06 or home_advantage >= home_upper - 1e-06)
    diagnostics = FitDiagnostics(approximation='Laplace approximation to the Gaussian random-effects Bradley-Terry marginal likelihood', converged=True, optimizer_message=str(best.message), season_count=len(records), game_count=sum((season.n_games for season in records)), outer_iterations=int(getattr(best, 'nit', 0)), objective_evaluations=evaluations, log_marginal_likelihood=-float(best.fun), maximum_mode_gradient=float(maximum_mode_gradient), tau_lower_bound=tau_lower, tau_upper_bound=tau_upper, tau_at_boundary=tau_at_boundary, home_advantage_at_boundary=home_at_boundary)
    return PopulationFit(home_advantage=home_advantage, tau=tau, diagnostics=diagnostics)

CUP_FINALS = {('2023-12-09', frozenset(('IND', 'LAL'))), ('2024-12-17', frozenset(('MIL', 'OKC')))}

@dataclass(frozen=True)
class Game:
    season: int
    date: str
    home: str
    away: str
    home_score: int
    away_score: int
    neutral: bool

def read_games(source) -> list[Game]:
    if isinstance(source, (str, Path)):
        with Path(source).open(newline='') as handle:
            return read_games(handle)
    games, seen = ([], set())
    for row in csv.DictReader(source):
        if row['playoff'] not in ('', 'NA'):
            continue
        neutral = row['neutral'] == '1'
        home, away = (row['team1'], row['team2'])
        if not (row['is_home'] == '1' or (neutral and home < away)):
            continue
        day = date.fromisoformat(row['date']).isoformat()
        if (day, frozenset((home, away))) in CUP_FINALS:
            continue
        season = int(row['season'])
        scores = (int(row['score1']), int(row['score2']))
        key = (season, day, frozenset((home, away)))
        if key in seen or home == away or min(scores) < 0 or (scores[0] == scores[1]):
            raise ValueError(f'invalid or duplicate game: {key}')
        seen.add(key)
        games.append(Game(season, day, home, away, *scores, neutral))
    return sorted(games, key=lambda g: (g.season, g.date, g.home, g.away))

def early_games(games: list[Game], cutoff: int) -> list[Game]:
    if cutoff < 1:
        raise ValueError('cutoff must be positive')
    counts, selected = (Counter(), [])
    for game in sorted(games, key=lambda g: (g.season, g.date, g.home, g.away)):
        home, away = ((game.season, game.home), (game.season, game.away))
        if counts[home] < cutoff and counts[away] < cutoff:
            selected.append(game)
        counts[home] += 1
        counts[away] += 1
    return selected

def group_seasons(games: list[Game]):
    groups = defaultdict(list)
    for game in games:
        groups[game.season].append(game)
    output = []
    for season, rows in sorted(groups.items()):
        teams = tuple(sorted({g.home for g in rows} | {g.away for g in rows}))
        index = {team: i for i, team in enumerate(teams)}
        output.append(SeasonGames(str(season), teams, np.array([index[g.home] for g in rows], dtype=int), np.array([index[g.away] for g in rows], dtype=int), np.array([g.home_score > g.away_score for g in rows], dtype=float), np.array([g.neutral for g in rows], dtype=bool)))
    return output

def pooled_calibration():
    source = ROOT / 'data'
    details = dict(
        line.split(': ', 1)
        for line in (source / 'source.txt').read_text().splitlines()
        if line.startswith(('Data file: ', 'Data SHA-256: '))
    )
    raw = source / details['Data file']
    if hashlib.sha256(raw.read_bytes()).hexdigest() != details['Data SHA-256']:
        raise ValueError('calibration data hash mismatch')
    games = [g for g in read_games(raw) if 2005 <= g.season <= 2025]
    seasons = group_seasons(early_games(games, 25))
    fit = fit_population(seasons)
    if fit.diagnostics.tau_at_boundary or fit.diagnostics.home_advantage_at_boundary:
        raise RuntimeError('pooled calibration reached boundary')
    return {'distribution': 'iid Gaussian log strengths', 'tau': fit.tau, 'home_effect_calibration': fit.home_advantage, 'home_effect_simulation': 0, 'estimator': fit.diagnostics.approximation, 'pooled_seasons': [s.season for s in seasons], 'season_count': len(seasons), 'game_count': sum((s.n_games for s in seasons)), 'cutoff': 'Both teams have fewer than 25 previous regular-season games', 'source_sha256': details['Data SHA-256'], 'era_split': False, 'scope': 'Early-season, not demonstrated tanking-free; pooled sample with fixed fitted hyperparameters.'}

if __name__ == '__main__':
    print(json.dumps(pooled_calibration(), indent=2))
