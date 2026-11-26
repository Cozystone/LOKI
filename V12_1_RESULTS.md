# V12.1 · real-body feedback and an I0 candidate

Claude's A2 report for the frozen v12 is the primary real-body evidence. In rooms 9101/9104 the bedrock one-step MAE was .0026/.0029 versus .0014 for holding. Current-corner accuracy was .604/.470, next-corner .445/.356, at 1.84/1.75 ms per body including journeys. With belief entering selection, median food waiting increased from 91 to 129 and from 251 to 264. Deaths were 7 vs 6 and 90 vs 91. These are supplied measurements from two rooms. They do not establish a causal confidence threshold for policy adoption. The mixed water/food first-journey score is unsuitable for a food-only verdict.

## Changes

Import LokiV121 from loki_memory.journey_121 for a separate shadow instance. Its constructor and bridge-facing methods follow v12. Retain separate checkpoint paths. Source pin 29e44dc061bed82326f9141fcf065ce0fc23d264. SHA-256 of the LF-normalized journey_121.py is 88e07d6dccba04e6b82ce89e34c1d98a9e6dcae7575ff8acfc4294ee011364c4. Its base journey.py remains the published v12 source with SHA-256 939ebcaa9e962b7dcd1458a57a36079b8d67b169e915e8e2d25ac2c5632108bd.

1. Each channel scores five residual multipliers (0, .125, .25, .5, 1) against subsequently visible outcomes using an exponentially weighted absolute error. Holding wins initial ties. Parameters never see the outcome when forecasting. Forecast and one-step imagine use the selected scale. The inherited multi-step journey readout remains a separate estimator.
2. Learn which anonymous motor components accompanied actual positive replenishment while remaining at a place. A visible, unfulfilled opportunity at the same place, with evidence of headroom and a previously productive command, softly discounts occupancy. Positive evidence clears the discount; it also fades with time. Unassigned mass is exposed when no alternative has been learned. This is evidence about availability conditional on interaction. A blocked/failed action, alias, or noisy reservoir can still confound the result.
3. For observed replenishment sites, combine empirical successor counts with one pseudo-observation weighted inversely by incoming transition frequency, excluding the previous site. This is an explicit one-to-one-cycle bias. No corner, period or motor identity is supplied. Sparse observations can skip multiple phases, so this is not guaranteed to recover the true sequence. No evidence crosses bodies.

## Frozen synthetic test

24 paired seeds on each schedule, 6,000 ticks per uninterrupted life, sparse unscheduled visits. All variants saw identical commands and observations. Each scheduled read was scored; empty predictions scored zero and tied winners received fractional credit. Chance over three corners is one third. The special probe used three sensory values, three motor options and an unambiguous place cue. A separate synthetic full-width test exercised 140 channels, 14 motor components and 30 place components. These fixtures do not reproduce real A2.

| Cyclic schedule | v12 | v12.1 | v12.1 without failed-opportunity evidence | v12.1 without successor prior |
|---|---:|---:|---:|---:|
| Current corner accuracy | .485 | .578 | .524 | .576 |
| Next corner accuracy | .415 | .458 | .458 | .415 |
| Mean old-site mass on 258 matched misses within 90 ticks of a move | .350 | .125 | .395 | .209 |

The candidate improved current accuracy in 20/24 paired seeds and next accuracy in 17/24. Both levels miss .8. On genuinely shuffled schedules, current accuracy was .523 vs .376 and next .349 vs .295. Past evidence supplied little information about the randomized future.

Cyclic slow-store MAE was .001091 vs v12 .002507 and hold .001154. The action-driven channel tied v12 at .0000414 versus hold .000550. For independent random increments, v12.1 .001000016 did not beat hold .001000000. R2's every-channel condition remains unmet. The four full-width synthetic lives gave mean MAE .05687 vs v12 .05691 and hold .11297. Mean of seed-wise p95 latency for forecast + observe + journeys was .348 ms vs .285 ms on this desktop, after JIT warm-up. New matrix size on this configuration is 188,832 bytes. These figures exclude rendering, runtime overhead outside the measured calls and power consumption.

## Claude handoff

Run v12.1 in I0 on the same single read and executed action as v12. Count training opportunities per body: distinct sites actually replenished, visible productive attempts, matched misses, and how many ordered pairs were observed. Score all eligible reads as well as supported reads. Compare time until old-corner mass falls and time until the **correct** new corner leads; leaving the old site does not establish knowing the new one. For journey ranking re-read the same checkpoint with food-only lack. Stratify one-step absolute error by channel and by the rare ticks when the reservoir really changes. Use sealed additional rooms before a B1 claim. A policy comparison should include zero imposed minimum weight and retain the same exploration budget.

A shared implementation, body schema or fixed hypothesis class transfers no observations. Each body retains its own learned parameters, place basis, phase, attempt history and experience chain. If the founder later authorizes social learning, another body's report needs its own source, age and empirically learned coordinate alignment; it is a distinct intervention against a no-sharing and shuffled-report control. No such mechanism is enabled here.

[Pre-registration](V12_1_PROTOCOL.md) · [Sealed per-seed results](measurements/v12_1_sealed.json) · [Full-width cost](measurements/v12_1_width.json)

Run python -m research.v12_1_sparse, python -m research.v12_1_width and python -m unittest discover from the repository root.
