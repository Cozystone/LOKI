# LOKI v11 · A finite predictive-state memory

An observation is a 24-component normalized measurement. Commands have four anonymous values. A goal is a sensory example. The evaluator knows the physical pose, the motor permutation and success condition. The learner receives observations, commands and consequences. The policy in this release is a research probe of the memory. ATANOR retains authority over deployed action selection.

## One filter for place and interaction

Up to 96 hypotheses share emission rows E and nonnegative weights C[a,i,j]. The same fit, predict and infer routines serve both objects and places. A source can have multiple successors, and several sources can emit the same observation. With reference scale s, the filter uses

    T[a,i,:] = normalize(C[a,i,:])
    prior = b @ T[a]
    L[j] = exp(-(MSE(o,E[j]) - min_cost)/s)
    posterior = normalize(L * (.998 * prior + .002/n))

A missing row defaults to staying in place. Observed return transitions can supply a missing row from a different action if empirical reverse agreement exceeds .75. No inverse actions are specified. For a completed probe, source smoothing multiplies an appearance likelihood by the predicted likelihood of the observed result. This conditions identity on the observed effect. Two sources with identical appearances and identical conditional futures still cannot be distinguished by that experiment.

With known incoming sensory history and contradictory outgoing evidence, a spare emission row can inherit the current appearance while part of the incoming support is redirected. An existing compatible clone is preferred. Conflicting evidence can also discount a previously confident transition. These are approximate operations. A single unusual event has no intrinsic tag saying whether a sensor failed or physics changed. At saturation a novel observation aliases to an existing emission. No unlimited episode archive or independent ground-truth identity is available.

## Continuous predictions and planning

For continuous inputs, each action trains an online ridge readout on a constant, the observation, the previous action and 24 sinusoidal features. A posterior-weighted next emission and this continuous residual prediction are mixed by local observation support. On cross-room transfer, a separate training calibration segment estimates the useful scale of the predicted delta. The readout has its own parameters. Online spatial planning uses the same stochastic topology as source inference, with a sensory goal likelihood and 32 Bellman backups. Transition entropy discounts risky actions. An uncertain belief can earn an expected information gain from distinguishing observable consequences. Episode-local action attempts discourage immediately repeating an unsuccessful strategy. A fresh observation corrects belief and a sufficiently surprising outcome triggers local model repair and renewed planning.

During acquisition, actions with least conditional exposure can be selected under the current belief. This controls which outcomes get measured within the same interaction budget. The reset sampler used for room acquisition supplies independently sampled starting observations. This constitutes stronger access than one continuously living body has.

## Space, computation and evidence

E is 96 by 24, C is 4 by 96 by 96, and four readouts have 53 by 24 coefficients and 53 by 53 inverse-covariance matrices. Additional fixed arrays store the basis, four scale estimates and episode-local action exposure. The serialized state uses 226,450 bytes. A checkpoint replaces the previous file atomically. A new episode begins with fresh visit penalties. Python objects, temporary working arrays, NumPy, simulator and OS memory are additional. Normalized transition matrices and planning work scale quadratically in active hypotheses. The 256 KiB measurement concerns model state, not peak inference RAM.

The ten-million-event path streams sufficient statistics for unique, exact emission anchors without adaptive continuous readout. It is equivalent to counting transitions in this restricted case. It does not exercise clone ambiguity, changing camera frames or the full planner ten million times. The key hypothesis is that a bounded store can preserve reusable conditional structure and revise it after contrary evidence. The filter, recursive least squares, Bellman planning and information gain are established mathematical techniques. The experimental contribution is their coupling and its measured limits.

## Persistent uncertainty in a nonlinear rollout

Revision 14 adds one 24-component disturbance direction u and a running lag-one cross moment c. On consecutive observations it updates c toward (u·e[t−1])(u·e[t]), with e the pre-update next-sensory residual. An Oja update aligns u with the observed residuals. For open-loop continuous prediction it transports two paths with equal weight:

    x[±,k+1] = clip(f(x[±,k], a[k], b[±,k]) ± u sqrt(max(c,0)))
    estimate[k] = (x[+,k] + x[−,k]) / 2

A persistent hidden disturbance passes through the nonlinearity before marginalization. Each path infers sensory hypotheses at its own predicted observation. This offers a testable difference from iterating the average forecast. It represents only one disturbance direction and two support points. Nonconsecutive observations do not contribute a new lag product. This is an engineering hypothesis about these dynamics, with a paired single-path ablation.

Revision 15 profiles multiplicative exposure and reduces the influence of the six largest component errors in source attribution. Revision 16 divides each Bellman continuation factor by 1 + .12 times the temporary visit count of that source/action. A costly cycle can then change the value of approaching it. Persistent checkpoint size for revisions 14–16 is 227,394 bytes; the 226,450-byte figure above applies to revision 13.
