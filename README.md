# Team Nexus — More than a Hack 2027, Phase 1: Ball Possession

**Submission repository.** Team Nexus's entry for [More than a Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027),
Phase 1: one label (`H` / `A` / `D` / `X`) for each of the 199,319 frames of the tactical video.

| | |
|---|---|
| Team | **team-nexus** |
| Submission file | `team-nexus.csv` — 199,319 rows, `frame,label` |
| Technical document | [`docs/technical-document.md`](docs/technical-document.md) |
| Method | evidential per-frame scores + a structured Viterbi decode; **no learned model, no training data, no video** |
| Public 5 minutes | **macroF1 0.8073 · spellF1 0.4941** (published floor `baseline_smooth50`: 0.6207 / 0.3256) |

## Reproduce the submission

```bash
pip install -r requirements-lock.txt          # python >=3.11 (measured on 3.14.7)
                                              # or `pip install -r requirements.txt` on 3.8+
export MTAH_DATA_DIR=/path/to/mtah2027        # the gated match bundle; see data/README.md

python -m pipeline.data.features                     #  7.6 s -> artifacts/features.npz
python -m pipeline.cli.submit --out team-nexus.csv   #  2.1 s -> the submission file
python evaluation/validate_submission.py team-nexus.csv
```

Deterministic: numpy only, no RNG, fixed feature ordering — the same bundle gives a byte-identical
file, `md5 4141029a8480a78454ede4906280f270`. The submitted file's md5 is recorded in §9 of the
technical document.

Check it and reproduce the evidence:

```bash
python evaluation/run_evaluation.py   # macroF1 / spellF1 on the 5 published minutes
python -m pipeline.cli.ablate         # the label-free ablation table (no labels exist there)
python -m pytest -q                   # 36 tests, no match data required, < 1 s
```

## What is ours

| path | what |
|---|---|
| `pipeline/data/` | readers (everything indexed to the video frame) and the 199-channel feature matrix |
| `pipeline/model/` | emissions and priors, Viterbi + structural cleanup, learned-emission experiments |
| `pipeline/checks/` | label-free diagnostics, the Smart-Tagging phase oracle, a video overlay renderer |
| `pipeline/cli/` | `submit`, `ablate` |
| `docs/technical-document.md` | the submitted technical document (approach, rules, training data, limitations, reproduction) |
| `docs/pipeline.md` | per-step measurements: what was tried, kept, and rejected |
| `docs/references.md` | literature review, and the design decision each source drove |
| `tests/` | 36 tests, no match data needed |

The match bundles are **not** in this repository — `data/` and `artifacts/` are gitignored, and
nothing here redistributes the gated assets or a reconstruction of them. Everything below the line
is the organisers' challenge repository (MIT): their task, data, metric, scorer and public ground
truth. We changed only three things down there — the reading list, the file listing and the
requirements section gained a line about our code — and marked them where they appear.

---

# More than a Hack 2027 — Phase 1: Ball Possession

**Organised by [Metrica Sports](https://metrica-sports.com), [FC Barcelona](https://www.fcbarcelona.com)
and the [Mobile World Capital Barcelona](https://www.mwcbarcelona.com)**, at the Talent Arena in Barcelona.

A football data challenge on real professional match data, open to analysts, data scientists and
engineers. You are given one complete LaLiga match — two camera feeds, automatic tracking of every
player and the ball, automatic tactical tagging, and the camera-to-pitch geometry — and one
question to answer with it.

---

## The task

**For every frame of the match, say who has the ball.**

```
frame,label
11185,A        away in possession, ball alive
11186,A
...
18240,D        dead ball
...
18684,H        home in possession, ball alive
```

That is the whole task. One CSV, one row per video frame, 199,319 rows. It is scored automatically
against a private frame-by-frame ground truth, so there is a number at the end and no argument
about it — no jury, and no presentation to argue from.

The tracking is automatic and uncorrected: it has gaps and identity
switches, and the ball itself is invisible on 41% of the frames you are scored on.

---

## The challenge in one screen

| |                                                                                                                           |
|---|---------------------------------------------------------------------------------------------------------------------------|
| **Input** | one full match: tactical video, broadcast video, automatic tracking **with the ball** for both, Smart Tagging, homography |
| **Output** | `frame,label` CSV, 199,319 rows, labels `H` / `A` / `D` / `X`                                                             |
| **Ranked on** | `spellF1` — how many of the match's real possessions you recovered as whole possessions                                   |
| **Also reported** | `macroF1` — the per-frame view, and the first tie-break                                                                   |
| **Baseline to beat** | `spellF1` **0.2433**, in this repository                                                                                  |
| **Ground truth you get** | the first **5 minutes** of play — to check your pipeline, not to train on                                              |
| **Ground truth you are scored on** | the full match, **private**                                                                                               |

It is deliberately **not** a tracking challenge. The tracking, tagging and homography are given to
you so that nobody has to rebuild detection, re-identification and camera calibration before
getting to the actual question. See [CHALLENGE.md](CHALLENGE.md).

---

## Try it right now

The scorer, a 5-minute ground truth and four working baselines are in this repository. No
enrolment, no download, no data:

```bash
pip install -r requirements.txt
python evaluation/run_evaluation.py
```

Scored on the public 5 minutes, not on the full match -- which is why the order below is not the
order of the floor quoted above:

```
scoreboard  gt/BAR-ATH_possession_gt_public.csv  199319 frames  7500 graded  21 gt segments
ground truth has 17 possessions of mean 14.2 s  <- compare the poss and dur columns

rank  submission                macroF1  spellF1   poss     dur  
----  ------------------------  -------  -------  -----  ------  
   1  baseline_smooth125         0.5812   0.4242     16    18.5  
   2  baseline_smooth50          0.6207   0.3256     26    10.8  
   3  baseline_smooth25          0.6200   0.2868     36     7.6  
   4  baseline_raw               0.5972   0.0378    131     2.0  
```

Those four are **the same algorithm four times** — nearest tracked player to the ball, then a
majority vote over a window. Only the window changes. `macroF1` barely moves; `spellF1` moves by a
factor of eleven. That disagreement is the whole point of the metric, and
[EVALUATION.md](EVALUATION.md) explains it.

Once you have the data, regenerate them yourself and go further:

```bash
python starter/baseline.py                              # rebuild the four baselines
python evaluation/validate_submission.py <your>.csv     # before every submission
```

**Validate every file before you submit it.** `validate_submission.py` runs exactly the checks the
official scorer runs, and a file that fails them is dropped from the board with a one-line reason.
It takes two seconds and it is the single most common way a good entry scores nothing.

---

## Getting the data

The match assets are released to enrolled participants who have accepted the terms of use.

1. Enrol: **[More than A Hack 2027](https://landing.mobileworldcapital.com/more-than-a-hack-2027)**
2. Read and sign the terms and conditions in the registration form
3. You receive a download link.
4. Unpack it into `./data`, or point `MTAH_DATA_DIR` at wherever you put it.

Full description of every asset and how the frames line up: **[DATA.md](DATA.md)**.

---

## Read these, in this order

| | |
|---|---|
| **[CHALLENGE.md](CHALLENGE.md)** | the problem, the four labels, what counts as possession |
| **[DATA.md](DATA.md)** | every provided asset, its format, and the frame alignment |
| **[EVALUATION.md](EVALUATION.md)** | how you are scored, and why — read before optimising anything |
| **[SUBMISSION.md](SUBMISSION.md)** | the file contract, the three deliverables, how to submit |
| **[RULES.md](RULES.md)** | what you may and may not do |
| **[FAQ.md](FAQ.md)** | |
| **[OVERVIEW.md](OVERVIEW.md)** | the competition as a whole — context, why the 2027 format changed, the calendar |

Then ours: **[docs/technical-document.md](docs/technical-document.md)** (the document we submit —
approach, rules implemented, training data, limitations, reproduction),
**[docs/pipeline.md](docs/pipeline.md)** (per-step measurements: what was tried, kept and rejected),
**[docs/references.md](docs/references.md)** (literature review and the decision each source drove).


---

## What is in this repository

```
games.py                    per-match constants, asset paths and the frame alignment
evaluation/
  possession_metrics.py     the metric library — every number on the board is defined here
  run_evaluation.py         the scorer. This exact file produces the official result
  validate_submission.py    check a file before you send it
  gt/                       the public 5-minute ground truth
  results/                  the four baselines; put your submissions here
starter/
  baseline.py               the reference baseline, numpy only
  load_atd.py               read the tracking and line it up with video frames
  load_smart_tagging.py     read the tactical tagging as video-frame intervals
  load_homography.py        read the homography; project pitch <-> pixels
  utils.py                  pitch plotting for the notebooks
  01_smart_tagging.ipynb    guided tour of the tactical tagging
  02_tracking_atd.ipynb     guided tour of the tracking data, with plots
  03_first_submission.ipynb data in, scored submission out — start here
data/                       unpack the downloaded match assets here. Empty in a fresh clone

pipeline/                   Team Nexus's submission pipeline (added here — see above)
  data/                     video-frame-indexed readers and the 199-channel feature matrix
  model/                    emissions and priors, Viterbi + structural cleanup
  checks/                   label-free diagnostics, phase oracle, video overlay renderer
  cli/                      `submit`, `ablate`
docs/                       technical document (submitted), per-step measurements, references
tests/                      36 tests, no match data needed
requirements-lock.txt       exact versions the reported numbers were measured on
```

`games.py`, `evaluation/`, `starter/` and `data/` are the organisers' starter repository, unchanged —
except `evaluation/reports/nexus/nexus_final.json` and `evaluation/reports/ablations/abl_final.json`,
two aggregate reports we committed as evidence for the technical document's tables.

The scorer you run locally is the scorer that decides the competition. If it scores on your
machine, it scores the same way on ours.

---

## Requirements

**Python 3.8 or newer.** The evaluation needs `numpy` and `pandas` and nothing else; it is tested
under 3.8 and 3.11 and produces identical output under both. `starter/baseline.py` needs nothing
more either — you can reach a scored submission without installing anything else.

The notebooks additionally want `kloppy` and `matplotlib` (`requirements-starter.txt`), and
**Python 3.9+ is recommended for those**: on Windows, installing Jupyter under 3.8 hits a
`pywinpty` build failure that has nothing to do with this repository.

The pipeline in `pipeline/` has the same two dependencies and no others — no torch, no sklearn, no
video decoding. The numbers in the technical document were measured on Python 3.14.7 with
`requirements-lock.txt`; on older interpreters `requirements.txt` is enough and produces the same
file.

---

## Timeline

| |                |
|---|----------------|
| registration opens | **October 1, 2026** |
| data released | **October 20, 2026** |
| submission deadline | **November 17, 2026** |
| results announced | **Before November 26, 2026**      |

---

## Questions

Open an issue for anything about the data, the metrics or the documentation — corrections are
genuinely useful, and a fix that helps you helps everyone equally. For enrolment and the terms of
use: **hackathon.talentarena@mobileworldcapital.com**.

## Licence and citation

The code and documentation in this repository are released under the MIT Licence — see
[LICENSE](LICENSE). **The match data is not covered by it**: it is separately licensed for this
competition only, under the terms you accept at enrolment.

```bibtex
@misc{mtah2027,
  title  = {More than a Hack 2027: Phase 1 - Ball Possession},
  author = {Metrica Sports and FC Barcelona Innovation Hub},
  year   = {2027},
  url    = {https://github.com/metrica-sports/more-than-a-hack-2027}
}
```
