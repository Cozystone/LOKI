# v13 · ATANOR place and episode storage

Going supplies observations, odometry when available, and experienced changes in unnamed body channels. The store returns evidence. Going owns goals, routes and action selection. No action vocabulary or environment identifier is an input to v13.

## The distance and what a confident match means

Let a trace carry appearance k, slowly varying sensory context c, and odometry p in chart F. With 32 key components and 16 context components the dimensionless cost is

    D = ||k-k_i||²/.32 + ||c-c_i||²/.1296 + I[registered in F_i] ||p+offset-p_i||²/.4225

Similarity is exp(-D/2). Confidence is the difference between the two highest similarities, up to numerical stabilization. Novelty is one minus the largest similarity. All three depend on what the sensors distinguish. A new physical object producing familiar evidence can score highly. Use normalized sensory features and calibrated local odometry units. FP16 storage rounds coordinates and features. Chart correction estimates a translation. Rotated axes, changed scale and a changed feature encoder need their own registration.

## Memory budget and exact retrieval

TraceStore keeps one fixed table and body-local charts. A slot has 32+16 sensory components, position, harm, channelwise experienced relief, visits, time, generation, eight outgoing links with counts and sleep values. Token = generation × capacity + slot. Sleep retires handles before reusing slots. A full awake table returns -1 for an unmatched new observation and counts pressure. Means and finite outgoing degree discard detail.

TraceAtlas adds an exact CPU search index. With a registered frame, spatial tiles have a known minimum possible distance to the query. Scan whole hash chains for neighboring tiles. Stop once the exact eighth cost beats the lower bound for every omitted tile. Hash collisions add work and do not exclude candidates. Without a spatial certificate, scan a contiguous weighted sensory projection. An upper bound from actual pilot candidates and a conservative floating-point allowance select rows for exact rescoring. A dense set of indistinguishable cues can still require every row. Initial registration depends on landmark uniqueness, checked with position omitted.

The hypothesis comes from the interaction of path integration and landmarks studied in the rat hippocampus. The hash index, metric and timing thresholds are engineering decisions. [Gothard, Skaggs and McNaughton (1996)](https://gothardlab.org/wp-content/uploads/publications/1996_Dynamics_of_Mismatch.pdf)

## The same life across domains

TraceAtlas accepts pos=None for audio, device telemetry, an internal vector or a digital event whose location is unknown. These traces occupy a reserved unlocated chart. They can be retrieved by key and context and linked to dated episodes. They cannot supply coordinates or serve as landmarks. A position-free read searches both placed and unplaced traces. A located observation can later be stored separately without writing coordinates into an old unlocated trace.

Inputs share a 32/16 feature contract. Each body supplies a compatible encoder and a causal context derived from preceding observations. A change of encoder or units must be reflected in the binding and tested. An arbitrary recoding across devices is an open alignment problem. A conversation can supply an acoustic or externally grounded event key. The store has no speech recognition, transcription or semantic interpreter.

A bounded event ring stores body, trace handle, interval and predecessor serial. Repeated observations at consecutive beats extend one interval. Missing observations, unaccepted writes, transfer or cut break the chain. A timestamp query reconstructs only surviving intervals. Losing an interval or recycling its underlying trace truncates the account. This permits two visits to the same junction to have different pasts. It also explains why capacity has no direct conversion into years. At 500 new intervals per day, 65,536 intervals cover about 131 days before overwrite, even if all corresponding places survive. This is arithmetic, not a workload measurement.

A separate four-way table counts observed triples. It can recall a different successor for the same junction depending on the incoming trace. Collisions and capacity limit rare continuations. Local role signatures combine measured edge diversity, entropy, neighbor branching and channel outcomes. They support simple structural analogies across appearances. Place, identity, language meaning and inferred causes are different questions. The signature is hand-designed; the topology and outcomes come from experience. Temporal context as a retrieval cue also has an established theoretical literature. [Howard and Kahana (2002)](https://sites.bu.edu/tcn/files/2015/12/HowardKahana02-JMP.pdf)

## Consolidation

The small store merges nearby same-chart traces, preserves maximum harm and relief across merges, rewires outgoing/incoming neighbors and removes stale neutral low-visit traces. Sleep propagates empirical channel changes backward on the observed, frequency-weighted graph. Each sweep uses value_i = max(relief_i, .97 × sum_j p_ij value_j). This is a summary of observed routes. The caller decides whether it matters. The cited reverse-replay experiment recorded awake rats after spatial experience. Choosing idle intervals for this computation is our own scheduling decision. [Foster and Wilson (2006)](https://pubmed.ncbi.nlm.nih.gov/16474382/)

Above 4,096 slots the atlas uses a rotating 512-row window. It merges eligible isolated neighbors within that window, protects linked vertices from deletion, and runs one local value sweep. A 65,536-slot pass takes 128 calls; values on longer paths can require several passes. The time budget is cooperative, without a hard real-time guarantee. Consolidation is lossy. No measurement establishes how much a lifetime of human experience would compress.

## Claude integration

Install with pip install ".[trace]". TraceStore supports batched torch CPU or CUDA. TraceAtlas provides the extended host implementation. A single authority must serialize calls on one store. Link once per body per beat, including an invalid predecessor immediately after a cut. Pass the same body-row mapping to cue, write, anchor and link. Snapshot-affecting operations run between ticks. Treat handles as opaque and check validity after sleep.

~~~python
from loki_memory import TraceStore, TraceAtlas

s = TraceStore(channels=C, bodies=64, device=device, binding=encoder_digest)
ids, similarity, confidence, novelty = s.cue(key, pos, ctx)
now = s.write(key, pos, ctx, rise, drop, clock, live)
s.link(previous, now, live)
evidence = s.read(ids)
report = s.consolidate(2000)
~~~

After a body changes call cut, query cue(..., landmark=True) and anchor(ids, confidence, pos_now). For single-writer laptop/glasses handoff, TraceVolume.create, checkpoint, transfer and resume operate on one authoritative volume. Resume increments the ownership epoch. Copying the database into two independent writable volumes forks the history. Atomic file save/load is available for offline checkpoints. Keep a previous binding if older bodies must remain readable.

For a serialized CPU tick, evidence, now = atlas.observe(key, pos, ctx, rise, drop, clock, live, previous) shares the pre-write ranking on the single-body placed path. A regression test compares the full snapshot with the separate cue/write/link sequence. For the larger CPU atlas, recall_episode(body, clock, steps) returns chronological (trace, first beat, last beat) tuples. continuations(first, second) returns observed successor IDs and counts. role(ids) returns unlabeled structural signatures. To integrate into ATANOR, run the place-store switch beside the 64-centroid reference using the same lives, the same adapter and no simulator identity. Check A2 life outcomes, C1/C7 looping and A2→A3→A2 return across at least three starts. The LOKI tests establish storage behavior. The ATANOR group owns the behavior trial.
