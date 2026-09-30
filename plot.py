#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import subprocess
import tempfile
from fractions import Fraction
from pathlib import Path


OUTPUT_NAMES = (
    "mmu-information-classes-population.pdf",
    "mmu-nba-rules-population.pdf",
)

PREAMBLE = r"""\documentclass[10pt,border={0pt 5pt 0pt 2pt}]{standalone}
\usepackage[T1]{fontenc}
\usepackage{lmodern,microtype,amsmath,pgfplots,pgfplotstable}
\usepgfplotslibrary{fillbetween}
\pgfplotsset{compat=1.18}
\definecolor{wfblue}{HTML}{176AA8}
\definecolor{ordgreen}{HTML}{277C69}
\definecolor{h2hred}{HTML}{B33C51}
\definecolor{nbaochre}{HTML}{A57511}
\definecolor{plotink}{HTML}{24303A}
\definecolor{plotgray}{HTML}{6C737B}
\pagestyle{empty}
\setlength{\parindent}{0pt}
\pgfplotsset{abstractaxis/.style={
 width=0.89\linewidth,height=2.82in,scale only axis,
 xmin=0,xmax=1,ymin=0,ymax=.75,
 xtick={0,.2,.4,.6,.8,1},ytick={0,.1,.2,.3,.4,.5,.6,.7},
 yticklabels={0\%,10\%,20\%,30\%,40\%,50\%,60\%,70\%},
 xlabel={Value of a win relative to the first pick, $\kappa$},
 ylabel={Probability of selecting the weakest team},
 axis lines=left,axis line style={plotgray!65},tick align=outside,
 ymajorgrids,grid style={black!10},
 tick label style={font=\sffamily\footnotesize},
 label style={font=\sffamily\footnotesize},
 legend style={font=\sffamily\scriptsize,draw=none,fill=none,cells={anchor=west}},
 every axis plot/.append style={no marks,line width=1.1pt},clip=false}}
\begin{document}
\begin{minipage}{7.2in}
\centering
"""

POSTAMBLE = r"""\end{minipage}
\end{document}
"""


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def load_results(results):
    results = Path(results)
    try:
        first = json.loads((results / "figure-1.json").read_text())
        second = json.loads((results / "figure-2.json").read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read graph results from {results}: {error}") from error

    _require(first.get("complete") is True, "figure-1.json is incomplete")
    _require(second.get("complete") is True, "figure-2.json is incomplete")
    _require(first.get("calibration_sha256") == second.get("calibration_sha256"),
             "result files use different calibrations")
    _require(first.get("n") == 30 and first.get("k") == 3,
             "figure 1 requires 30 teams and three games per pair")
    _require(second.get("n") == 30 and second.get("games_per_team") == 82,
             "figure 2 requires 30 teams and 82 games per team")
    _require(first.get("smoothing") is False and second.get("smoothing") is False,
             "result files must contain unsmoothed curve values")
    _require(first.get("mmu") and first.get("ordinal", {}).get("curve") and
             first.get("full_information", {}).get("curve"),
             "figure 1 is missing a plotted series")
    _require(second.get("mmu") and set(second.get("rules", {})) == {"current", "upcoming"},
             "figure 2 is missing a plotted series or NBA rule")
    return first, second


def rule_annotations(second):
    annotations = {}
    for name in ("current", "upcoming"):
        rule = second["rules"][name]
        threshold = float(Fraction(rule["ic_threshold"]))
        match = next(
            (row for row in second["mmu"] if abs(row["kappa"] - threshold) < 1e-14),
            None,
        )
        _require(match is not None, f"MMU curve omits the {name} IC threshold")
        baseline = rule["estimate"]["accuracy"]
        gain = match["accuracy"] - baseline
        _require(abs(gain - rule["gain_at_ic_threshold"]["accuracy"]) < 1e-12,
                 f"saved {name} gain does not match the plotted values")
        annotations[name] = {
            "threshold": threshold,
            "nba_accuracy": baseline,
            "mmu_accuracy": match["accuracy"],
            "gain_pp": 100 * gain,
        }
    return annotations


def _write_table(path, rows, columns):
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(header for header, _ in columns)
        writer.writerows([
            format(float(row[key]), ".17g") for _, key in columns
        ] for row in rows)


def _figure_one(first):
    free = format(float(first["unconstrained"]["accuracy"]), ".17g")
    body = rf"""\pgfplotstableread[col sep=comma]{{figure-1-mmu.csv}}\mmudata
\pgfplotstableread[col sep=comma]{{figure-1-ordinal.csv}}\ordinaldata
\pgfplotstableread[col sep=comma]{{figure-1-full.csv}}\fulldata

{{\sffamily\bfseries\color{{plotink}}Accuracy across information classes\par}}
{{\sffamily\small\color{{plotink}}Approximate NBA setup\par}}
\vspace{{5pt}}
\begin{{tikzpicture}}
\begin{{axis}}[abstractaxis,legend pos=south east]
\addplot[name path=wfupper,draw=none,forget plot] table[x=x,y expr=\thisrow{{wf}}+1.96*\thisrow{{ws}}]{{\mmudata}};
\addplot[name path=wflower,draw=none,forget plot] table[x=x,y expr=\thisrow{{wf}}-1.96*\thisrow{{ws}}]{{\mmudata}};
\addplot[wfblue!14,draw=none,forget plot] fill between[of=wfupper and wflower];
\addplot[wfblue,forget plot] table[x=x,y=wf]{{\mmudata}};
\addplot[name path=ordupper,draw=none,forget plot] table[x=x,y expr=\thisrow{{ordinal}}+1.96*\thisrow{{os}}]{{\ordinaldata}};
\addplot[name path=ordlower,draw=none,forget plot] table[x=x,y expr=\thisrow{{ordinal}}-1.96*\thisrow{{os}}]{{\ordinaldata}};
\addplot[ordgreen!12,draw=none,forget plot] fill between[of=ordupper and ordlower];
\addplot[ordgreen,forget plot] table[x=x,y=ordinal]{{\ordinaldata}};
\addplot[h2hred,forget plot] table[x=x,y=h2h]{{\fulldata}};
\addplot[plotgray,dashed,forget plot] coordinates {{(0,{free})(1,{free})}};
\addlegendimage{{plotgray,dashed}}\addlegendentry{{Unconstrained accuracy}}
\addlegendimage{{h2hred}}\addlegendentry{{Full-information upper bound}}
\addlegendimage{{wfblue}}\addlegendentry{{MaxMin Utility}}
\addlegendimage{{ordgreen}}\addlegendentry{{Ordinal upper bound}}
\end{{axis}}
\end{{tikzpicture}}
"""
    return PREAMBLE + body + POSTAMBLE


def _nba_rule_lines(name, values):
    color, label, side, delta = {
        "current": ("h2hred", "Current NBA rule (through 2026)", "right", 0.095),
        "upcoming": ("nbaochre", "Upcoming NBA rule (2027-2029)", "left", -0.032),
    }[name]
    threshold = format(values["threshold"], ".15g")
    baseline = format(values["nba_accuracy"], ".15g")
    mmu = format(values["mmu_accuracy"], ".15g")
    callout_y = format(values["nba_accuracy"] + delta, ".15g")
    return rf"""\addplot[{color},densely dotted,forget plot] coordinates {{(0,{baseline})({threshold},{baseline})}};
\addplot[{color}] coordinates {{({threshold},{baseline})(1,{baseline})}};
\addlegendentry{{{label}}}
\addplot[{color},only marks,mark=*,mark size=1.5pt,forget plot] coordinates {{({threshold},{baseline})}};
\node[anchor=south east,font=\sffamily\scriptsize,text={color}] at (axis cs:.99,{baseline}) {{{100 * values['nba_accuracy']:.2f}\%}};
\draw[{color},line width=.4pt] (axis cs:{threshold},{baseline}) -- (axis cs:.225,{callout_y});
\node[anchor=west,font=\sffamily\scriptsize,text={color}] at (axis cs:.23,{callout_y}) {{IC from {values['threshold']:.4f}}};
\draw[<->,{color},thick] (axis cs:{threshold},{baseline}) -- node[midway,{side},rotate=90,font=\scriptsize,fill=white,inner sep=1pt] {{$\sim${values['gain_pp']:.1f} pp}} (axis cs:{threshold},{mmu});"""


def _figure_two(second):
    rules = rule_annotations(second)
    rule_lines = "\n".join(_nba_rule_lines(name, rules[name])
                           for name in ("current", "upcoming"))
    body = rf"""\pgfplotstableread[col sep=comma]{{figure-2-mmu.csv}}\mmudata

{{\sffamily\bfseries\color{{plotink}}Accuracy of MaxMin Utility and NBA Draft Rules\par}}
{{\sffamily\small\color{{plotink}}Approximate NBA setup\par}}
\vspace{{5pt}}
\begin{{tikzpicture}}
\begin{{axis}}[abstractaxis,legend style={{at={{(1,.55)}},anchor=east}}]
\addplot[name path=wfupper,draw=none,forget plot] table[x=x,y expr=\thisrow{{wf}}+1.96*\thisrow{{ws}}]{{\mmudata}};
\addplot[name path=wflower,draw=none,forget plot] table[x=x,y expr=\thisrow{{wf}}-1.96*\thisrow{{ws}}]{{\mmudata}};
\addplot[wfblue!14,draw=none,forget plot] fill between[of=wfupper and wflower];
\addplot[wfblue] table[x=x,y=wf]{{\mmudata}};
\addlegendentry{{MaxMin Utility}}
{rule_lines}
\addlegendimage{{plotgray,densely dotted,line width=1.1pt}}\addlegendentry{{Dotted: not incentive compatible}}
\end{{axis}}
\end{{tikzpicture}}
"""
    return PREAMBLE + body + POSTAMBLE


def prepare_sources(results, workdir):
    first, second = load_results(results)
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)

    paths = {}
    tables = (
        ("figure-1-mmu.csv", first["mmu"], (("x", "kappa"), ("wf", "accuracy"), ("ws", "se"))),
        ("figure-1-ordinal.csv", first["ordinal"]["curve"], (("x", "kappa"), ("ordinal", "ordinal"), ("os", "ordinal_se"))),
        ("figure-1-full.csv", first["full_information"]["curve"], (("x", "kappa"), ("h2h", "capped_estimate"))),
        ("figure-2-mmu.csv", second["mmu"], (("x", "kappa"), ("wf", "accuracy"), ("ws", "se"))),
    )
    for filename, rows, columns in tables:
        path = workdir / filename
        _write_table(path, rows, columns)
        paths[filename] = path

    sources = (
        ("mmu-information-classes-population.tex", _figure_one(first)),
        ("mmu-nba-rules-population.tex", _figure_two(second)),
    )
    for filename, source in sources:
        path = workdir / filename
        path.write_text(source)
        paths[filename] = path
    return paths


def render(results, output):
    executable = shutil.which("pdflatex")
    if executable is None:
        raise RuntimeError("pdflatex with PGFPlots is required to render the figures")

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".plot-build-", dir=output) as temporary:
        workdir = Path(temporary)
        sources = prepare_sources(results, workdir)
        built = []
        for output_name in OUTPUT_NAMES:
            source = sources[Path(output_name).with_suffix(".tex").name]
            run = subprocess.run(
                [executable, "-interaction=nonstopmode", "-halt-on-error", source.name],
                cwd=workdir,
                capture_output=True,
                text=True,
            )
            if run.returncode:
                raise RuntimeError(f"pdflatex failed for {source.name}:\n{run.stdout[-4000:]}")
            if "Overfull" in run.stdout:
                raise RuntimeError(f"pdflatex reported an overfull box for {source.name}")
            built.append((source.with_suffix(".pdf"), output / output_name))
        for source, target in built:
            shutil.copyfile(source, target)
    return tuple(target for _, target in built)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Render the two MMU population graph PDFs.")
    parser.add_argument("--results", type=Path, default=Path("build/recomputed"),
                        help="directory containing figure-1.json and figure-2.json")
    parser.add_argument("--output", type=Path, default=Path("figures/output"),
                        help="directory for the two rendered PDFs")
    args = parser.parse_args(argv)
    for path in render(args.results, args.output):
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
