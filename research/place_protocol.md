# SH2c transport pilot (before running)
One fixed visible bearing/depth association at a time. Encode bearing times positive
radial depth; fit all motion coefficients from consecutive sightings. No rotation
matrix, camera parameters, map or action scores in learner. This assumes depth is
proportional to range; an actual eye may violate it.
24 independent seeds (2000..2023), 200 visible pairs per body, 32 new held-out
16-step occlusions each. Same trajectories for learned transport, true yaw only
and wrong-body motion. Score every endpoint (including unsupported); require
>=80% within 45 degrees and improvement over both controls. Report clean,
5% relative depth noise and squared depth separately, no tuning on results.
Real room, power, and calibrated confidence are outside this pilot.
