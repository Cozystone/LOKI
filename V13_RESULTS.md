# v13 measurement record

Research pin for the core and held-out harness is **70dbfc5**. The mixed-domain implementation was frozen at **37cb8a4**. The combined tick is **018527b**. The support prior, numerical thresholds and held-out IDs were fixed before the core test. Raw per-seed metrics are linked below.

## Held-out place retrieval

Across seeds 26101–26124, retrieval of each of 64, 128, 256, 512 and 1,024 distinguishable grid places scored 1.000. Each trial used smoothed binary observations from the independent v11 falsification generator, small sensory/odometry perturbations and consistent metric coordinates. The test enumerates the places; autonomous exploration was outside the experiment. Baseline A, A after B, and A after sleep were all 1.000. B also retained 1.000. Each world contained 144 places, and B had the same current appearance field as A but a deliberately separated preceding ambient signal. The two-context task is easy when the context field is available. These results measure retrieval. Earlier v11 capacity numbers included planning and cannot serve as a direct speed or accuracy baseline. [24 raw lives](measurements/v13_sealed.json)

In every seed an ambiguous post-reset cue failed the anchor test, then a distinct landmark at the second beat recovered the chart translation with zero measured error on the chosen integer displacement. Sleep merged one deliberately near duplicate and pruned two old neutral records. Three slots were freed in each trial. The directional sleep diagnostic recovered the appropriate channel on both observed predecessor arms. [Core protocol](V13_PROTOCOL.md)

**Confidence fails for an unseen but observationally indistinguishable place.** On the preregistered strong ambient shift, novelty AUROC was 1.000 and high-confidence false acceptance was zero with full known coverage. In eight further seeds, moving the ambient context by only 0.02 or 0.05 gave high-confidence false acceptance of 100%. Identical appearance and history at distinct coordinates scored 100% with odometry and 50% after removing it. Similarity is evidence about this representation, with no independent identity guarantee. [Sensitivity data](measurements/v13_indistinguishable.json)

## Order and abstraction

A shared junction had two equally frequent outgoing edges. Its predecessor distinguished both continuations (2/2 vs a best unordered choice of 1/2). Role matching across two separately textured but isomorphic graphs got 5/5, compared with 27.3% mean accuracy under category permutation. This is a tiny constructed example of transferring measured graph structure. The role features are designed, and the experiment supplies no evidence of discovering linguistic concepts. The chronology tests separately cover interleaved bodies, returning to a junction twice, holding a trace for multiple beats, expiry, recycled handles and a mixed-domain checkpoint. [Order and role controls](research/v13_acceptance.py) · [Chronology, crash and ownership tests](test_trace_store.py)

One atlas in each of twelve synthetic lives stored spatial visits alongside audio spectra, slow body-like signals and binary symbol-like patterns. The two members of each pair had the same current key and different preceding observations. With previous context, each family scored 100%. Removing it gave 49.0%, 50.0% and 50.0%, respectively. All 12 instances recovered their complete 146-event dated sequence after serialization. The judge supplies synthetic histories from informative distributions. Only normalized features and prior context enter the organ. Actual dialogue, cross-device alignment and everyday recall remain unmeasured. [Multidomain protocol](V13_DOMAIN_PROTOCOL.md) · [Results](measurements/v13_domains.json)

## Reproduction

Use Python 3.10+ and install the trace extra. CPU benchmark code pins PyTorch to one thread. For CUDA benchmarking use an appropriate CUDA-enabled PyTorch build and matching NVRTC on the desktop. All tests ran on Windows. NVRTC compiles the small kernels on first use. Latency is warm, and excludes installation, first compilation, transfer, snapshotting and sleep.

~~~sh
python -m pip install ".[trace]"
python -m research.v13_acceptance --sealed
python -m research.v13_scale
python -m research.v13_latency
python -m research.v13_domains
python -m research.v13_alias
python -m unittest -q test_trace_store
~~~

The original sealed run used --worlds pointing at falsify_v11/worlds.py in independent commit a0fead793ddeb1e7ef212a9ceb51bdc8b59e72a5 (local file SHA-256 5ce1472e1d4b66c25ecff15129b395035dfee5608245c8348c9c21eb98e43aee). The default reproducer includes the observation-only fixture and matched the independent source bit-for-bit on the 1,024-place development seed. Repeated runs of the published seeds are regressions.

## Populated scale and latency

Ryzen 9 9950X3D, one CPU thread. Each size was filled through the public write API in grid order and in a random permutation. 128 queries per geometry condition and order were scored against an independent exact sort of every stored row. The combined tick was measured on a separate 256-query sample. The samples differ and the separate-call benchmark runs immediately after a dense reference scan. Differences between timing columns cannot all be attributed to fusion.

| Slots | Order | Fixed arrays MiB | Registered cue p95 ms | Cold cue p95 ms | Separate cue/write/link p95 ms | Combined observe p95 ms | Top eight agreement, warm/cold |
|---:|---|---:|---:|---:|---:|---:|---|
| 8,192 | grid | 4.73 | 0.297 | 0.355 | 0.676 | 0.422 | 1.000 / 1.000 |
| 8,192 | shuffled | 4.73 | 0.294 | 0.311 | 0.648 | 0.352 | 1.000 / 1.000 |
| 65,536 | grid | 32.56 | 0.930 | 1.595 | 1.944 | 0.686 | 1.000 / 1.000 |
| 65,536 | shuffled | 32.56 | 0.747 | 1.742 | 1.506 | 0.542 | 1.000 / 1.000 |

At 65,536 places the registered scorer inspected a median 160–176 candidates. Cold recall touches every compact sensory projection. The original gather-based cold path took 15.64–17.47 ms at that size; the revision retained exact reranking. The cold and separate-call 1 ms targets were missed at full scale. These are two admission orders of one generated field with exact revisit keys. Route diversity, camera changes and device performance need separate data. Each sleep slice covered 512 of the 65,536 rows. [Initial scale run](measurements/v13_scale_initial.json) · [Final timings and occupancy](measurements/v13_scale.json)

For the base 1,024-slot store, five warmed repetitions each measured 120 timed ticks per mode. Per-run p95 ranges in milliseconds follow.

| Hardware / batch | Hit | Allocation | Full, rejected |
|---|---:|---:|---:|
| CPU / 1 | 0.520–0.552 | 0.303–0.318 | 0.530–0.537 |
| RTX 5080 / 64 | 1.022–1.969 | 1.177–1.529 | 1.087–1.538 |

The base workload met the 1 ms CPU and 2 ms GPU gates in all five final repetitions. Earlier GPU pressure p95 reached 2.334 ms. GPU runs synchronize before and after the complete tick. These timing tests use prepared fixtures. [Five runs](measurements/v13_latency.json) · [Previous miss](measurements/v13_latency_before_boundary_fusion.json)

## Device scope

The 65,536-slot state has 20,709,571 serialized bytes including temporal records. Derived search structures raise the fixed total to 34,144,643 bytes. Temporary arrays, snapshot copies, Python, the tensor library and the allocator add working memory. A separate fresh process on this Windows build measured **526.5 MiB resident** with the initialized atlas pages touched, of which **492.9 MiB** was already resident after library import. This is an important constraint on the glasses deployment. Measured milliseconds and bytes do not establish watts or battery life. [Process measurements](measurements/v13_runtime.json)

TraceVolume tests used one authoritative local database with two simulated host identities. A stale laptop checkpoint was rejected; a glasses observation survived the return; the device boundary added no artificial edge. A process killed immediately before replacing a snapshot left the earlier snapshot readable. Real laptop, Rokid, ATANOR A2 life, C1/C7 looping and S0/S1 return experiments remain open. The storage experiment has no human control.
