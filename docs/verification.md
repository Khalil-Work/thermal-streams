# Verification

The test suite (`tests/`, 770 test items, `pytest -q`) establishes the
properties below. Expected values inside the tests are computed through
independent code paths (plain `math.exp` arithmetic, analytic limits,
or a hand-written RK4 integrator), not by calling the implementation
under test.

## What each test establishes

### (a) Energy balance closes to roundoff — `tests/test_energy_balance.py`

For all five arrangements across a sweep of NTU (0.3 to 10), C_r
(0.15 to 1) and either stream carrying C_min (160 configurations), the
heat lost by the hot stream equals the heat gained by the cold stream:

- `energy_balance_residual_W == 0` to within 1e-9 W absolute and 1e-13
  relative,
- both per-stream duties equal the reported duty to 1e-13 relative,
- outlet temperatures are physical (hot cools, cold heats, positive
  terminal differences).

Result: the outlet-temperature computation is internally consistent with
the effectiveness relation used. Largest residual observed in the
published sweep (3000 cases, NTU up to 100): **3.5e-10 W** (3.7e-14
relative) — `docs/results/run_summary.json`.

### (b) Effectiveness limits — `tests/test_effectiveness_relations.py`

- **Counter flow, NTU -> infinity:** 1 - eps matches the analytic
  asymptotic gap (1 - C_r) e^{-NTU(1-C_r)} / [1 - C_r e^{-NTU(1-C_r)}]
  to 1e-9 relative, and eps == 1 to 1e-15 at NTU = 200, C_r = 0.01.
- **Parallel flow:** eps(NTU = 300) equals the exact saturation value
  1/(1 + C_r) to 1e-12 relative for C_r in {0, 0.25, 0.5, 0.75, 1}, and
  eps stays strictly below it at finite NTU.
- **Equal capacity rates:** the counter-flow C_r = 1 branch equals
  NTU/(1 + NTU) exactly and blends continuously with the general branch.
- **Additional arrangements:** shell-and-tube and both crossflow-mixed
  variants match their NTU -> infinity limits
  (2 / (1 + C_r + sqrt(1 + C_r^2)), (1 - e^{-C_r})/C_r, 1 - e^{-1/C_r})
  to 1e-12.
- Every arrangement is bounded below by parallel flow and above by
  counter flow (thermodynamic ordering of arrangements), satisfies
  0 < eps <= min(1, NTU), and shares the universal small-NTU expansion
  eps = NTU [1 - (1 + C_r) NTU / 2] to third order.

### (c) C_r = 0 anchor — `tests/test_effectiveness_relations.py`

Every arrangement reproduces eps = 1 - exp(-NTU) exactly at C_r = 0
(1e-14 relative), and the full solver reproduces it with a numerically
huge opposite stream (C_max = 1e12 W/K, C_r = 1.2e-9) to O(C_r) as
expected. The constant-temperature stream stays at its inlet value to
Q/C_max (about 1e-7 K in the test). Published sweep agreement
(300 cases, NTU 0.1-50): max deviation **1.2e-9**, consistent with the
C_r = 1.2e-9 offset of the idealized stream.

### (d) LMTD cross-check — `tests/test_lmtd.py` (strongest check)

For the same outlet temperatures, the duty from eps-NTU equals the duty
from the independent LMTD formulation Q = U A dT_lm, where dT_lm is
computed from the terminal differences with a `log1p`-based
logarithmic-mean evaluation:

- **Counter flow** (required check): 11 C_r levels (0.05 to 1.0,
  including C_r = 1 where dT_1 = dT_2 and the LMTD degenerates) x 2
  C_min-side assignments x 24 NTU values per decade-scaled sweep up to
  NTU = 15.8 — agreement better than 1e-10 relative everywhere. Maximum
  observed in the published sweep (528 cases): **2.8e-11** relative.
- **Parallel flow** (extra): the identity is analytically exact, but the
  float evaluation of the parallel outlet difference
  dT_in e^{-NTU(1+C_r)} loses digits through cancellation between two
  ~300 K temperatures once it approaches the ~1e-14 K roundoff floor.
  The sweep is therefore restricted to the well-conditioned range
  NTU <= 5, where agreement holds to 1e-10 (maximum observed: 1.9e-12).
- The LMTD helper itself is pinned against closed values (equal
  differences; the 2:1 ratio dT_lm = dT/ln 2) and rejects non-positive
  or non-finite arguments.

### (e) Monotonicity — `tests/test_effectiveness_relations.py`

eps is strictly increasing in NTU on a 300-point sweep (NTU = 0.05 to 15)
for every arrangement and C_r in {0, 0.3, 0.7, 1}, and non-decreasing on
an extended sweep to NTU = 100 (where eps saturates at its analytic
bound at float resolution).

### Profiles — `tests/test_profiles.py`

The analytic counter-flow profiles are verified three ways:

1. endpoints reproduce the rating solution's outlet temperatures to
   1e-12 K (both capacity orderings and the equal-capacity-rate case);
2. the incremental energy balance
   C_h (T_h,in - T_h(s)) = C_c (T_c,out - T_c(s)) holds at every station
   to 1e-11 relative;
3. an independent hand-written RK4 integration (4000 steps) of the
   governing ODEs, started from the hot inlet and the cold outlet,
   reproduces the profiles to 1e-9 K *and lands on the far-end boundary
   conditions* (hot outlet, cold inlet) to 1e-9 K.

Also: the local temperature difference decays exponentially with transfer
area, stream temperatures stay monotone and inside the inlet/outlet
envelope, and non-counter arrangements are rejected.

### Conditioning regressions — `tests/test_conditioning.py`

The 29 added test cases cover the reproduced failure and adjacent contracts:

- Fifteen full-field comparisons against an **80-digit Decimal reference**:
  NTU in {4,50,100}, C_h/C_c assignments {1200/2000, 2000/1200,
  1500/1500} and both orientations of a 1e-10 relative capacity difference.
  Each field has 101 stations. Maximum recorded deviation is **5.7e-14 K**;
  the test threshold is 2e-10 K. Endpoints, monotonicity, boundedness and
  incremental energy balance are also checked.
- Independent 10,000-step RK4 integration at NTU=100 for both capacity
  assignments. It starts at the hot-inlet end when C_h<C_c and at the
  cold-inlet end when C_h>C_c, so perturbations decay. Agreement threshold
  is 2e-8 K. Integrating the growing direction would amplify roundoff and
  give a misleading reference failure.
- Three near-C_r=1 effectiveness checks against Decimal arithmetic, five
  symmetric wide-scale LMTD cases (including 1e-300/1e300 and near 1e308),
  normalization of float-convertible config inputs, and three explicit
  errors for nonrepresentable derived UA/NTU/duty.

The old forward implementation is retained only in the example reference
module for comparison. On the same 15 cases its maximum error is **65 K**.
`conditioning_metrics.json` and `conditioning_profiles.csv` retain the
measured before/after values. The plot's error-axis display floor is
1e-14 K; raw zeros are preserved in the CSV. These are parametrized
equation checks, not 770 distinct physical experiments.

### (f) Input validation — `tests/test_validation.py`

Clear `ValueError`s (naming the parameter and the offending value) for
non-positive or non-finite U, A, capacity rates and temperatures,
non-positive inlet temperature difference, unknown arrangement names
(error lists the valid options), NTU <= 0 / non-finite, C_r outside
[0, 1] (scalars and arrays), and non-positive LMTD arguments.

## Published artifacts

`examples/run_verification.py` regenerates everything in
`docs/results/`: the eps-NTU cover figure, the documented-case
counter-flow profile figure, a 48-frame animation of the profiles
growing with NTU (1000 x 600 px, nominal 8 fps), a 250-row heat-balance
CSV over the full sweep, and `run_summary.json` with runtime versions and
all metrics above. GIF timing is quantized to 120 ms per frame.

`animation_profiles.csv` saves all 11,568 station samples underlying the
48-frame animation (frame, NTU, s, hot K, cold K). The animation is a
steady area/NTU sweep, not a time-dependent startup solution. The new
`conditioning_comparison.png` and its CSV/JSON compare all-field errors
with the obsolete evaluation path and 80-digit reference. Figures and
animation use white interiors without grids or shaded background fills.

## Limits of the verification

- No pressure drop, no fouling, no axial conduction, no ambient heat
  loss: these are absent from the model, so no test can say anything
  about them.
- No phase change is modelled except the C_r = 0 idealization (one
  stream at constant temperature); real condensers/evaporators deviate.
- Constant properties and uniform U are assumed throughout; no
  temperature-dependent property variation, no partial-flow
  maldistribution.
- Configurations are limited to the five implemented arrangements; in
  particular the LMTD route is only exact for parallel and counter flow
  (no F-factor method for shell-and-tube or crossflow).
- Floating-point conditioning still bounds terminal-LMTD checks: the
  parallel-flow sweep is restricted to NTU <= 5, and counter flow to
  NTU <= 15.8. Rounded outlet differences may become zero at more extreme
  NTU, causing the positive-argument LMTD helper to reject them. The stable
  profile evaluator remains bounded; it does not restore lost terminal
  differences or establish physical model accuracy beyond these assumptions.
