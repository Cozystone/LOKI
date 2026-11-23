# LOKI v11 · What the bounded model can do

**Revision 16 met the ten numerical Level 3 gates in these simulators.** The independent confirmation changed the maze generation rule and camera palette, the object-camera perturbations, the alias-task order and the delayed plant. Both 24-room cohorts used a 32-action limit. This result concerns a memory-controlled agent in measured worlds. Tests of identity, plants and rooms trained separate instances of the same learner.

| Gate | Final sealed cohort | Independent confirmation |
|---|---:|---:|
| Synthetic identity precision / recall | .888 / .888 | .854 / .854 |
| Appearance-only precision / recall | .440 / .440 | .440 / .440 |
| 8-step dynamics MSE, v11 / v9 / v10 | .03433 / .05807 / .07006 | .01972 / .02374 / .06152 |
| 8-step matched recurrent / matched RLS | .04709 / .04207 | .02582 / .02693 |
| Held-out pooled next-view MSE, v11 / persistence | .01838 / .02284 | .00879 / .01173 |
| Physical return, unchanged / +90° / −90° / 180° | 22 / 21 / 21 / 22 of 24 | 24 / 24 in every condition |
| Physical return, displaced / partially masked | 21 / 21 of 24 | 24 / 24 of 24 |
| Novel goal, v11 / v10 / random, **identical transitions and endpoints** | 21 / 9 / 2 of 24 | 23 / 3 / 2 of 24 |
| Blocked route, eight-state cyclic detour | 24/24 | 24/24 |
| Ambiguous junction, active / passive | 1.000 / .625 | 1.000 / .667 |
| Novel goal, observation correction disabled | 15/24 | 15/24 |

The single model snapshot measured **227,394 bytes** at 10k, 100k, 1M, 2.5M, 5M, 7.5M and 10M actual events. At both 1M and 10M a once-observed action still selected its unique reachable consequence with predicted mass 1. Deleting its edge changed the chosen action. V10 also retained this isolated exception exactly at 1M. The generator repeated at most 11 distinct observable anchors; count statistics are sufficient. Under the sequence A→B→A→C the three observable contexts had zero final prediction MSE. Hidden, unobservable regime changes are harder.

## What produced the gains

- **Propagating a persistent disturbance through the dynamics helped.** On a separate fixed 96-plant cohort the two-path model scored .01986 at eight steps. Removing the two-path transport gave .02134. The paired seed bootstrap 95% interval for their difference was [−.00170, −.00127]. Corresponding differences vs v9, matched recurrent and matched random-feature RLS were all negative throughout their intervals. The small ordinary linear RLS scored .02069 in this replication. On the 24-plant changed-law cohort its difference interval included zero. Baselines include an echo-state recurrent predictor with a trained RLS readout; this does not compare every recurrent architecture. Full 1/4/8/16-step and FIFO, reservoir, VQ and prototype-only results are in the raw measurements.
- **Correction changed physical outcomes.** Against open-loop propagation, the paired novel-goal outcomes changed from failure to success six times on the primary cohort and eight times on the confirmation, with no reverse changes (two-sided exact sign p=.03125 and .00781). The active probe improved nine and eight seed-level junction outcomes with no reverse changes. The ring detour still succeeded with local law revision disabled, but mean actions rose from 8 to 18.875 on the primary cohort. Removing a bridge in the original tree disconnects the goal; the detour test therefore uses a loop with a real alternative path.
- **The planner can succeed for the wrong explanation.** On the candidate seeds propagating the temporary visit penalty raised novel-goal success from 20 to 21. The earlier local penalty did most of the work on this cohort. Robust source smoothing actually reduced identity from .901 to .888 versus ordinary smoothing on the same observations. The strong identity gain is from the observed action consequence. Assignment is closed-set, one winner per query, hence equal micro precision and recall. Primary false merge .0075 and false split .112 use different denominators. Some twins share *both* their camera distribution and action law. Even a judge knowing those distributions has a primary expected accuracy ceiling of .919 from this equivalence alone.
- **Acquisition matters.** The simulator independently resets initial poses during the 1,000 local interaction samples and supplies one continuous guide route. Goals outside that route require composition. The main comparator sees exactly the transitions acquired by v11. Revision 13 on its own seeds reached 19 novel goals with exposure-directed acquisition, 18 with random acquisition and 16 with two-step fragments. V9 has a recorded-route reversal API without a novel-goal argument. Its experienced-route baseline is in the suite file; it has no directly comparable novel-goal score.

## Diagnostics and counterexamples

- Initial ambiguous junction entropy averaged .6931 nats. Sequence-conditioned mass on the designed latent alternative averaged .99975. On unrelated appearance with the same noiseless probe law, consequence MSE fell from .04745 to numerical zero. These are controlled tests of sensory contingency. The identity probe result itself carries nearly exact law information.
- In an unseen room, the separate four-step sensory test gave .02395 MSE versus .02734 for persistence. A supplementary native 16×16 RGB test using revision 13's shared sensory readout lifted each predicted tile delta onto its original pixels. Pixel MSE was .02311 versus .02773. The unpooled part of the image is carried over. The evaluator-only endpoint Brier score was .0452. Four candidate actions averaged .0693 nats of predicted entropy and .206 posterior-weighted agreement with the observed next cue in this foreign-room diagnostic.
- The primary controller recovered physically after unannounced +90°, −90°, 180° and within-map displacement in 20, 21, 22 and 21 of 24 runs. Confirmation gave 23, 23, 23 and 24. A separate frozen-topology kidnapping probe raised a surprise flag in all 24 foreign rooms. It never localized to a true pose in that **unmapped** room. There is no defined cross-room path back in this simulator.
- Primary novel successes took paths with mean shortest-path/actual-length ratio .896 (confirmation .856). All three primary failures had an observed physical path but exhausted 32 decisions. They triggered 2, 10 and 3 local repairs; good one-step surprise scores can coexist with navigation failure. Mean occupied hypotheses were 87.0 of 96. The plots show the trajectories the agent actually took. [Three sealed failures](examples/v11_failures_16.png)
- Calibration fails under distribution shift. In the high-confidence bin on familiar dynamics, observed exact-emission hits were 8,200/8,418. In a different textured room, 0/8,752 high-confidence predictions matched the exact new emission. Continuous forecasts can improve MSE while categorical confidence is meaningless there. The current score should not be handed to ATANOR as certainty.
- In a separate revision 13 stress test, 192 one-shot cues mapped to independent binary consequences. Recall was .859 for the first half and .699 for the second. Once all 96 hypotheses are in use, new inputs collide. There is no evidence of indefinite faithful episodic recall, a human-like memory, or petabyte capacity.

## Resource accounting and execution

The candidate owns 226,428 bytes of numerical arrays, including 147,456 bytes of stochastic transition counts, 44,944 bytes of readout precision, 20,352 bytes of readout coefficients, 9,216 bytes of emissions, 2,688 bytes of fixed basis, 1,536 bytes of transient visit weights, 32 bytes of calibration and 204 bytes for persistent residual transport. Snapshot metadata and ZIP headers account for the different serialized size.

The optional Numba controller streams dense rows and caches at most eight nonzero outcomes per row. Rows with more successors are recomputed exactly. Its largest temporary is 15,360 bytes of weights and indices together; the remaining numerical decision buffers occupy less than 12 KiB at 96 hypotheses. The agent arrays plus these planning buffers fit 256 KiB. The compact kernel repeated the novel-goal scores of 21/24 and 23/24. Warm desktop decision latency at 96 hypotheses measured 0.127 ms median, 0.223 ms p95 (100 timed calls). This is one planning call. Python, JIT initialization, process RSS, rendering, flash I/O, and power are outside those figures. The NumPy reference creates dense scratch matrices and uses more RAM. An on-device full perception/learning loop on Rokid remains unmeasured.

## Research history and reproduction

The initial revision reached 5/24 at 180° and 8/24 novel goals. Revisions 1–13 investigated repair, contextual splits, reverse evidence, risk and acquisition. Revision 13 passed the initial numerical point estimates; its expanded changed-plant replication did not establish a significant v9 advantage. Revision 14 improved continuous rollouts but missed primary identity (.823) and novel navigation (17/24). Revision 15 missed both (.828 and 16/24). Revision 16 is the frozen reported candidate. The use of many independently seeded revisions is adaptive research; the fresh compositional holdout and separate plant replication matter more than the best earlier score. Intermediate iterations used world screens before complete retrospective suite runs. They were not all full prospective gate evaluations. Masked-view repair and normalization of inferred inverse rows were corrected, and affected final measurements rerun. No thresholds were relaxed.

Use Python 3.10+ from the repository root, install the research extra, and set the BLAS thread count to one. The tests and following modules require no simulator service or device.

~~~sh
python -m pip install -e ".[research-v11]"
python -m unittest discover
python -m research.v11_world 16
python -m research.v11_suite 16
python -m research.v11_dynamics 16
python -m research.v11_holdout 16
python -m research.v11_replication 16
python -m research.v11_capacity 16
python -m research.v11_matched_world 16
python -m research.v11_ablations 16
python -m research.v11_falsification
python -m research.v11_calibration
python -m research.v11_compact
python -m research.v11_report 16
python -m research.v11_gate
~~~

[Hypotheses](V11_PROTOCOL.md) · [Independent holdout](V11_HOLDOUT_PROTOCOL.md) · [Theory and equations](V11_THEORY.md) · [Per-cohort statistics and paired intervals](measurements/v11_summary_16.json) · [Sealed world traces](measurements/v11_world_16.json) · [Independent raw evidence](measurements/v11_holdout_16.json) · [Matched navigation baselines](measurements/v11_matched_world_16.json) · [Capacity](measurements/v11_capacity_16.json) · [Compact controller evidence](measurements/v11_compact.json)
