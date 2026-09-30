# MaxMin Utility: graph reproduction

Data and code for the two accuracy graphs in *From Ranks to Records*.

## Run

Use Python 3.11 or newer. PDF rendering also needs `pdflatex` with PGFPlots,
standalone, Latin Modern and microtype (available in standard TeX distributions).
Tested with Python 3.13.7 and the pinned dependencies below.

```sh
python -m venv .venv
source .venv/bin/activate
python -m pip install numpy==2.3.3 scipy==1.16.2
python simulate.py --output build/recomputed
python plot.py --results build/recomputed --output build/figures
```

This refits the strengths, generates the numerical results locally, and draws
both graphs. Generated results are not included in the repository.

The full run uses 300,000 simulated leagues and 10,000 additional strength
profiles for the full-information bound. Seeds are fixed in the code and saved
with the results. Use a new output directory for each simulation run.

## Method

`calibrate.py` fits a Bradley-Terry model to 7,674 early-season games from
2004-05 through 2024-25. A game is included only when neither team has already
played 25 regular-season games. Team log strengths have a normal distribution
with fitted standard deviation 0.7031772846. The fit uses a Laplace approximation
to the marginal likelihood and estimates home advantage separately. Early-season
selection does not establish that the data are entirely free of tanking.

`simulate.py` draws fresh strengths for 30 teams in each simulated league. Both
figures use these same strength draws and neutral-game Bradley-Terry outcomes.
Accuracy is the average draft-pick probability assigned to the genuinely weakest
team. MMU equalizes shifted utilities subject to nonnegative probabilities;
the NBA approximations assign probabilities by rank and average tied positions.

Figure 1 uses three games per pair, or 87 games per team. `bounds.py` computes
the ordinal bound from the smallest feasible win gaps for each ranking and tie
pattern. It computes the full-information donor bound for each strength profile,
averages it, and caps it by unconstrained accuracy. MMU is the record-based optimum
in this balanced model. Figure 2 uses an 82-game schedule approximation with
two, three or four games per pair. Its NBA rules are simplified single-pick
rank lotteries, not complete simulations of eligibility, Play-In or traded picks;
MMU optimality is not claimed for that schedule.

`plot.py` draws the generated numerical results and the gain at each NBA rule's
incentive-compatibility threshold. Expected accuracies and gains are Monte Carlo
estimates, not guaranteed numerical minima. Shading measures pointwise simulation
uncertainty conditional on the fitted distribution, not uncertainty in the fit.

## Data and sources

`data/games.csv` contains the source game-result rows for the 21 relevant seasons.
The fitting code excludes the two NBA Cup finals and applies the early-season
filter. It does not use Elo ratings. The source is
[Neil Paine's NBA Elo archive](https://github.com/Neil-Paine-1/NBA-elo/tree/37db89481b02cbdf5f7d0bc1e41271ebb11bd283).
The pinned version, checksum, selection details and full upstream MIT license
notice are in `data/source.txt`.

Numerical results can vary slightly across library versions and platforms.

The NBA-rule approximations follow the NBA's
[current lottery description](https://www.nba.com/news/nba-draft-lottery-explainer)
and [new lottery announcement](https://www.nba.com/news/nba-board-governors-approve-new-draft-lottery-system).
No affiliation or endorsement is implied. The upstream data notice does not
license this project's original code; an original-code license has not yet been assigned.
