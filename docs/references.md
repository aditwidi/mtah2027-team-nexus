# What the literature said, and what we did with it

Background reading for **More than a Hack 2027**, frame-level ball possession (Metrica Sports ×
FC Barcelona), compiled while the pipeline was being built. This is not a survey. It is a record of
which published results changed our design, which ones we checked against this dataset and found
wanting, and how confident we are in each number we quote.

Every figure below is tagged. `[full text]` means we read it in the paper. `[abstract]` means the
publisher's abstract page. `[summary]` means a search provider's synthesis of a paywalled manuscript,
and those should be treated as second-hand. `[local]` means we measured it on this dataset ourselves
and it is reproducible from the repository. Two of the three papers that shaped the design are in the
third group, so read §9 before quoting anything from here.

---

## 1. There is one paper on this exact task

Borghesi, Dusty Costa, Morra and Lamberti (2023), *Using Temporal Convolutional Networks to Estimate
Ball Possession in Soccer Games*, Expert Systems with Applications 234:119780. Its output space is
ours: a per-frame label of home possession, away possession, or dead ball, from optical tracking. It
reports 86.2% possession accuracy and 89.2% on dead/alive. `[abstract]`

The headline numbers are not what we used. The **input ablation** is. `[summary]`

| input to the classifier | three-class state | alive / dead |
|---|---|---|
| ball only | 58.35% | 83.62% |
| the five players closest to the ball | 79.41% | — |
| the full model | 86.2% | 88.25–89.2% |

Read those two columns together and the design of the challenge becomes obvious. The ball is a good
dead-ball sensor and a poor possession sensor. The neighbourhood of players around the ball is the
reverse. A single decision rule fed by one input, which is what the published baseline in this
repository is, loses most of the available signal on *both* heads for different reasons.

That is where our two-head split came from: dead and possession are scored independently, from
different evidence, and only meet in the decoder.

Two other details from the paper mattered. It feeds explicit **missing-data indicators** as features,
which is the right instinct when the ball is absent on 41% of the graded frames here. And its
sequence model aggregates players in a permutation-invariant way, which is the only workable choice
with 46 anonymous track slots that swap identity.

Caveats the authors state themselves, and we believe them: the dataset is a selection of
high-quality matches with manual labels, the train/validation/test split is by match and team, and the
rule-based competitors were reimplemented with inherited hyperparameters, so the +7.3 and +33.2
point margins are not a clean comparison.

## 2. Dead ball is a different problem, and it does not need the ball

Lang, Wild, Isenko and Link (2022), *Predicting the in-game status in soccer with machine learning
using player movement data*, Scientific Reports 12:6843, classifies active versus interrupted from
25 fps tracking of all 22 players with **no ball data at all**. It reports 92.0% frame accuracy and
93.1% F1 across 47 test matches from the Bundesliga. `[summary]`

Three things in it were worth more to us than the accuracy figure:

- **Multi-scale time shifts.** The feature set is position, velocity and acceleration at seven past
  and seven future shifts spanning about 0.4 s to 80 s, roughly 1,320 features. That is the cheapest
  feature win in the literature and it is exactly what our context group does.
- **A median filter over the output.** Their 3.64 s median filter extracts stoppage intervals, and
  stoppages under 2 s are excluded from evaluation. Published practice therefore agrees with the
  challenge's argument that a segmentation layer beats per-frame cleverness.
- **A realistic boundary error of about 1.7–1.8 s.** This one changed our expectations rather than our
  code. A 1.5 s error on each edge of a 14 s possession leaves IoU around 0.80; on a 4 s possession it
  is fatal. The consequence is that short possessions have to be protected, and frame-exact edges are
  not worth chasing.

Their model is tabular AdaBoost, not a deep network, which is worth noting for anyone assuming this
head needs sequence models.

## 3. The adjacent work we deliberately did not follow

**Kim et al. (2023), Ball Radar** `[abstract]` reconstructs ball trajectories from the multi-agent
configuration with a Set Transformer and a hierarchical Bi-LSTM, predicting possession as an
intermediate task. It is the cleanest published statement that player context can stand in for an
absent ball. We did not train it — the cost is far beyond the available time and the transfer to
Metrica's automatic tracking is unproven — but it is the reason our missing-ball features are built
around player geometry rather than around interpolating the ball harder.

**Peral, Capellera, Rubio, Agudo et al. (2025, 2026)** recognise the ball possessor from video, with
player tubes and temporal models, evaluated on 36 La Liga and Serie A matches with event timing
accurate to about 0.6 s. The WACV 2026 paper works on tactical camera footage, which is the feed we
have. This is the road not taken, and the honest reason is calendar: detection, tracking and tube
association are two or more days of infrastructure before the first possession label exists, and they
lead back to a signal we already hold numerically. It also sets a useful reference point — 0.6 s event
timing is what video-level supervision buys, so our 1.5 s tracking-only boundary error is respectable
rather than embarrassing.

**Vidal-Codina et al. (2022)** is the deterministic tracking-only family our published baseline
belongs to: a possession step that decides who has the ball per frame, then event rules on top. It is
the "rule-based state of the art" that the learning papers measure themselves against, which is a
reminder of how much headroom sits inside the rules themselves before any model is trained.

**Link and Hoernig (2017)** gave us vocabulary rather than method. Their taxonomy separates individual
ball control from a single touch, and individual possession from team possession. That is precisely the
ambiguity in `CHALLENGE.md` ("a touch is not a change of possession"; "controlled possession" as either
a few seconds of retention or two completed passes), and it is why our rules speak in terms of
sustained neighbourhood evidence and a minimum dwell rather than in terms of who touched the ball last.

**PathCRF** appeared in searching as ball-free possession inference from player trajectories. We could
not extract its body text, nothing here depends on it, and we list it only so that a reader who finds
it knows we saw it and could not verify it.

## 4. What no paper solves for this competition

The literature optimises frame accuracy or frame F1. None of these papers reports possession
segmentation quality, possession count error, or one-to-one matching of possessions. This competition
ranks on exactly that, which means the published state of the art can rank poorly here and the score
is made in the decoding layer rather than in the classifier. The challenge's own baselines demonstrate
it: the same algorithm four times, differing only in a smoothing window, moves `spellF1` from 0.0378 to
0.4242 while `macroF1` barely moves.

The other gap is label provenance. Those papers had vendor labels or manual annotations from dozens of
matches. Here the full-match truth is private and the five public minutes are for verification, so a
supervised emission would have to be trained on other games or on our own pseudo-labels. Both are
allowed. Both carry risk. We tested the second and it lost (§7).

And their dead ball is easier than ours. Vendor-labelled stoppages, clean definitions, roughly 40%
dead, high-quality optical tracking. Ours is derived from corrected tracking of a feed that is
explicitly automatic and uncorrected, in which the ball is missing on 41% of the frames we are scored
on. Their 89.2% is not a transferable expectation; the honest reference is this repository's floor.

---

## 5. The six decisions the reading produced

**Split the decision into two heads.** Separate dead/alive from home/away, with independent evidence,
combined in the decoder. This is the single largest structural decision in the project and it comes
directly from the Borghesi ablation. The published baseline, by contrast, decides `D` from ball absence
and home/away from the single nearest player, which is each head using the wrong signal.

**Score ownership over a neighbourhood, not over the nearest player.** For each team, the mean weight
of its three nearest players to the ball, in metres on a 105.3 × 68.0 pitch. The representation is
invariant to how many track slots a team happens to own, which turned out to matter: this match has 24
home slots and 22 away, and a naive sum over slots gave home a structural advantage that showed up as a
predicted home share of 0.709 against a truth of 0.573.

**Build dead ball from positive evidence.** Ball absence is one feature among several, not a rule:
gap length, an out-of-pitch ball, set-piece intervals, stationary-ball duration, player stillness,
half boundaries. The reason is measured rather than argued — treating absence as sufficient caps `D`
recall at 0.21 on this data.

**Use acausal multi-scale context.** Position, velocity, visibility and ownership at log-spaced lags
from 0.4 s to 64 s, in both directions. Lang's feature set is the template; the acausality is free
because the submission is scored offline.

**Decode with duration constraints.** Viterbi over `{H, A, D}` with asymmetric switch costs, then a
minimum-dwell cleanup. `CHALLENGE.md` settles the part that is easy to get wrong: a stoppage between
two runs of the same team keeps them separate, so cleanup must never merge a dead stretch.

**Do not tune on the five public minutes.** `EVALUATION.md` shows the window picking the wrong
smoothing variant for a one-parameter baseline, and the possession count and mean duration published
for that window is precisely the quantity one would fit. Our constants come from football priors and
from label-free structure over the whole match. Where we did use the window, the submitted document
says so in full.

**Ruled out from the start**, on evidence and budget: video-based possessor recognition, end-to-end
ball reconstruction, and the broadcast feed — unsynchronised to the tactical video, with usable
homography on 65.8% of its frames, and requiring an invented alignment.

---

## 6. What this dataset told us that no paper could

These are ours, measured with `pipeline/checks/diag.py` and on the public window. The design leans on
them harder than on anything published. `[local]`

| finding | measurement | what it drove |
|---|---|---|
| ball coordinates outside the pitch | 27.6% of dead frames, 0.0% of live ones | a near-pure `D` signal, worth 4.0 log-odds |
| mean tracked speed | 1.43 m/s dead, 2.38 m/s live | movement is the dead-ball signal that survives a lost ball |
| share of players above 3 m/s | 8.7% dead, 30.6% live | the strongest single movement feature |
| ball absent | 33.6% of dead frames, 6.4% of live ones | absence is evidence, far from sufficient |
| set-piece intervals | 15.9% of the window, 45.4% of its dead frames | restarts are where dead ball concentrates |
| track slots | 24 home, 22 away | an unnormalised sum over slots biases ownership toward home |
| team centroid nearer the ball, ball absent | the *defending* team, 384 of 384 | the formation term enters with a negative sign |
| set piece + visible ball still for 4 s | P(dead) = 0.830, base rate 0.196 | the largest single gain in the project, found by looking at the video |

The last two rows are the ones we would underline. Both are counter-intuitive, both were measured
before any code changed, and the second was found by rendering a disagreement with the ground truth and
watching it, not by reading.

---

## 7. The second pass: duration models and learned combinations

Halfway through, we went back to the literature to ask whether a learned or hybrid model could beat
hand-set priors. The answer was consistent and narrow.

Semi-Markov models of football sequences beat memoryless baselines because sports states are not
memoryless, which says the constant switch penalty is the wrong shape in principle and a hidden
semi-Markov decoder is the right instrument. Descriptions of production pipelines are hybrid by
construction — rules for explicit, well-defined events, learning for ambiguous classification, then
post-processing for consistency — which is the shape we already had. Self-training for scarce labels
reduces annotation need but propagates the teacher's errors. And the SoccerNet game-state task
predicts positions, roles, team affiliation and jersey numbers, but not possession, so there was no
public model to borrow.

We then tested the two actionable claims on this data instead of assuming them, and both failed to
improve the ranking metric.

**A strongly predictive feature can still be harmful.** Team spread separates possession at 0.840
against a 0.612 base rate with the ball visible, with the same sign when it is absent. Added as an
additive logit it raised `macroF1` from 0.7666 to 0.7856 and collapsed the window's possession count
from 18 to 12, dropping `spellF1` to 0.4000. A per-frame feature that is *too* convincing makes the
decoder stop switching.

**Self-training cannot beat its teacher.** A gradient-boosted emission trained out-of-fold on our own
pseudo-labels agrees with the rules 92.8% of the time, which is a genuinely useful check that the
features generalise across halves. Every mixture weight still degraded `spellF1` (0.4800 → 0.4000 at
the time of the test), because it learns the per-frame decisions without the structural decoding.

The conclusion we drew is the one that shaped the last day of work: the remaining headroom is in
sequence structure, and the specific unbuilt instrument is an explicit duration model.

---

## 8. How the reading maps onto the submitted document

`SUBMISSION.md` asks for six things. This file supplies the citations for four of them: the approach
and why (§1, §2, §5), the assets considered and dropped (§3), the training-data position (§4), and the
possession rules implemented (§3 and the Link & Hoernig vocabulary). The limitations section in the
submitted document draws on §4 and §9 here. Reproduction belongs to the repository, not to this file.

---

## 9. Evidence quality, stated plainly

| claim | status | note |
|---|---|---|
| Borghesi: 86.2% possession, 89.2% dead/alive | `[abstract]` | publisher and institutional PDFs returned 403; figures from the DOI abstract page |
| Borghesi input ablation: 58.35% / 79.41% / 83.62% | `[summary]` | second-hand synthesis of the accepted manuscript; **this is the number our two-head design rests on**, so it should be re-read from the PDF before anyone builds on it |
| Borghesi: team, role and missing-data inputs; permutation-invariant aggregation | `[summary]` | as above |
| Lang: 92.0% accuracy, 93.1% F1, 1,320 features, 15 lags, 3.64 s filter | `[summary]` | from the nature.com article and Table 1; the 1,320 = 22 × 4 × 15 arithmetic is our inference from the stated count, not a quoted formula |
| Lang: 1.8 s / 1.7 s start and end timing error | `[summary]` | used only to calibrate our expectations, not in any decision rule |
| Kim et al.: Set Transformer + hierarchical Bi-LSTM, possession as an intermediate task | `[abstract]` | no metrics were retrievable |
| Peral et al.: 36 matches, ~0.6 s event timing | `[summary]` | two papers conflated during retrieval; the VISAPP 2025 entry is possessor + events, the WACV 2026 entry is ball trajectory, ball state and possessor from tactical footage |
| PathCRF | unverified | body text not extractable; nothing here depends on it |
| every `[local]` number in §6 | measured here | reproducible with `pipeline/checks/diag.py`, `pipeline/checks/verify.py` and the public window |

The honest summary: the design rests on a second-hand ablation table and on a set of measurements we
made ourselves. That is not ideal, and it is also why the local measurements carry the weight in the
submitted document. Where the two disagreed, we trusted the local one — that is how the formation sign
and the set-piece-still term were found, and both of them beat what the literature suggested.

---

## 10. Sources

- Borghesi M., Dusty Costa L., Morra L., Lamberti F. (2023). *Using Temporal Convolutional Networks to
  Estimate Ball Possession in Soccer Games.* Expert Systems with Applications 234:119780.
  [doi:10.1016/j.eswa.2023.119780](https://doi.org/10.1016/j.eswa.2023.119780)
- Lang S., Wild R., Isenko A., Link D. (2022). *Predicting the in-game status in soccer with machine
  learning using player movement data.* Scientific Reports 12:6843.
  [doi:10.1038/s41598-022-19948-1](https://doi.org/10.1038/s41598-022-19948-1)
- Kim H. et al. (2023). *Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set
  Transformer and Hierarchical Bi-LSTM.* [arXiv:2306.08206](https://arxiv.org/abs/2306.08206) ·
  [code](https://github.com/hyunsungkim-ds/ballradar)
- Peral M., Capellera G., Rubio A., Agudo A. et al. (2025). *Temporally Accurate Events Detection
  Through Ball Possessor Recognition in Soccer.* VISAPP 2025.
  [PDF](https://www.scitepress.org/Papers/2025/133177/133177.pdf)
- Peral M., Capellera G., Ferraz L., Rubio A., Agudo A. (2026). *Multi-Modal Soccer Scene Analysis
  with Masked Pre-Training.* WACV 2026.
  [PDF](https://openaccess.thecvf.com/content/WACV2026/papers/Peral_Multi-Modal_Soccer_Scene_Analysis_with_Masked_Pre-Training_WACV_2026_paper.pdf)
- Vidal-Codina et al. (2022). *Automatic event detection in football using tracking data.*
  [arXiv:2202.00804](https://arxiv.org/abs/2202.00804)
- Link D., Hoernig M. (2017). *Individual ball possession in soccer.* PLOS ONE 12(7):e0179953.
  [doi:10.1371/journal.pone.0179953](https://doi.org/10.1371/journal.pone.0179953)
- (unverified) *PathCRF: Ball-Free Soccer Event Detection via Possession Path Inference from Player
  Trajectories.* [arXiv:2602.12080v2](https://arxiv.org/html/2602.12080v2)

Authoritative for scoring, and locally readable: [CHALLENGE.md](../CHALLENGE.md),
[DATA.md](../DATA.md), [EVALUATION.md](../EVALUATION.md), `evaluation/possession_metrics.py`.

```bibtex
@article{borghesi2023tcn,
  title   = {Using temporal convolutional networks to estimate ball possession in soccer games},
  author  = {Borghesi, Matteo and Dusty Costa, Lorenzo and Morra, Lia and Lamberti, Fabrizio},
  journal = {Expert Systems with Applications},
  volume  = {234},
  pages   = {119780},
  year    = {2023},
  doi     = {10.1016/j.eswa.2023.119780}
}

@article{lang2022ingame,
  title   = {Predicting the in-game status in soccer with machine learning using player movement data},
  author  = {Lang, Steffen and Wild, Raphael and Isenko, Alexander and Link, Daniel},
  journal = {Scientific Reports},
  volume  = {12},
  pages   = {6843},
  year    = {2022},
  doi     = {10.1038/s41598-022-19948-1}
}

@misc{kim2023ballradar,
  title         = {Ball Trajectory Inference from Multi-Agent Sports Contexts Using Set Transformer and Hierarchical Bi-LSTM},
  author        = {Kim, Hyunsung and others},
  year          = {2023},
  eprint        = {2306.08206},
  archivePrefix = {arXiv}
}

@inproceedings{peral2025possessor,
  title     = {Temporally Accurate Events Detection Through Ball Possessor Recognition in Soccer},
  author    = {Peral, Marc and Capellera, Guillem and Rubio, Antonio and Agudo, Antonio},
  booktitle = {Proceedings of VISAPP},
  year      = {2025}
}

@misc{peral2026multimodal,
  title     = {Multi-Modal Soccer Scene Analysis with Masked Pre-Training},
  author    = {Peral, Marc and Capellera, Guillem and Ferraz, Luis and Rubio, Antonio and Agudo, Antonio},
  booktitle = {WACV},
  year      = {2026}
}

@misc{vidaldina2022events,
  title         = {Automatic event detection in football using tracking data},
  year          = {2022},
  eprint        = {2202.00804},
  archivePrefix = {arXiv}
}

@article{link2017possession,
  title   = {Individual ball possession in soccer},
  author  = {Link, Daniel and Hoernig, Martin},
  journal = {PLOS ONE},
  volume  = {12},
  number  = {7},
  pages   = {e0179953},
  year    = {2017},
  doi     = {10.1371/journal.pone.0179953}
}
```
