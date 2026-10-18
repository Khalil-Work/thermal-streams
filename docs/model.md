# Model

Thermal Streams analyses heat exchangers with the effectiveness-NTU
(eps-NTU) method and provides energy-balance diagnostics and a logarithmic mean
temperature difference (LMTD) cross-check for supported arrangements.

## Assumptions (all relations below)

- Steady state, one-dimensional flow; both streams in single phase unless
  the C_r = 0 idealization is used.
- Uniform overall heat-transfer coefficient U over the whole area A.
- Constant capacity rates C = m_dot * c_p (constant properties).
- No heat loss to ambient, no axial conduction, no fouling (U is the
  clean, overall value), no pressure-drop effects.

## Definitions

| Quantity | Definition | Units |
|---|---|---|
| Duty | Q = heat transferred hot -> cold | W |
| Capacity rate | C = m_dot * c_p, per stream | W/K |
| Minimum / maximum capacity rate | C_min = min(C_h, C_c), C_max = max(C_h, C_c) | W/K |
| Capacity-rate ratio | C_r = C_min / C_max | dimensionless, 0 <= C_r <= 1 |
| Number of transfer units | NTU = U A / C_min | dimensionless |
| Inlet temperature difference | dT_in = T_h,in - T_c,in > 0 | K |
| Maximum possible duty | Q_max = C_min * dT_in | W |
| Effectiveness | eps = Q / Q_max | dimensionless, 0 < eps <= 1 |

Rating solution (all arrangements):

```
Q        = eps * C_min * (T_h,in - T_c,in)
T_h,out  = T_h,in - Q / C_h        (energy balance, hot stream)
T_c,out  = T_c,in + Q / C_c        (energy balance, cold stream)
```

## Implemented effectiveness relations

Each relation is stated exactly as implemented, with its validity domain.
All hold for NTU > 0 and 0 <= C_r <= 1.

### 1. Parallel (co-current) flow — `flow="parallel"`

```
eps = [1 - exp(-NTU (1 + C_r))] / (1 + C_r)
```

Valid for both streams entering at the same end. As NTU -> infinity the
effectiveness saturates at the exact bound eps_max = 1 / (1 + C_r).

### 2. Counter (counter-current) flow — `flow="counter"`

For C_r < 1:

```
eps = [1 - exp(-NTU (1 - C_r))] / [1 - C_r * exp(-NTU (1 - C_r))]
```

For C_r = 1 (equal capacity rates) the expression above is a 0/0 limit
whose closed form is used instead:

```
eps = NTU / (1 + NTU)
```

Numerically, let delta = 1−C_r and E = −expm1(−NTU*delta).
For unequal capacities the expression is evaluated as E/(delta+C_r*E),
avoiding cancellation in 1−C_r*exp(−NTU*delta). The equal-capacity limit
is used only at C_r=1 exactly; nearby unequal cases are not flattened to
an equality-tolerance plateau. As NTU -> infinity, eps -> 1 for every C_r < 1, with the asymptotic gap
1 - eps = (1 - C_r) e^{-NTU(1-C_r)} / [1 - C_r e^{-NTU(1-C_r)}].

### 3. C_r = 0 anchor: condenser / evaporator idealization

When one stream changes phase at constant temperature its effective
capacity rate is unbounded, C_r = C_min / C_max -> 0. Both the parallel
and the counter relation reduce exactly to

```
eps = 1 - exp(-NTU)
```

This is used as a correctness anchor for the whole code base: every
implemented arrangement must reproduce it at C_r = 0 (and the solver must
reproduce it with a numerically huge C_max). It is an idealization: real
condensers/evaporators have finite (though large) capacity rates and
pressure drop, neither of which is modelled.

### 4. Shell-and-tube, one shell pass (TEMA 1-2 type) — `flow="shell_and_tube"`

One shell pass with 2, 4, ... tube passes:

```
a   = NTU * sqrt(1 + C_r^2)
eps = 2 / [1 + C_r + sqrt(1 + C_r^2) * (1 + exp(-a)) / (1 - exp(-a))]
```

(the ratio (1+e^-a)/(1-e^-a) equals coth(a/2), which is how it is
evaluated). Valid for the ideal 1-shell-pass arrangement with an even
number of tube passes; as NTU -> infinity,
eps -> 2 / (1 + C_r + sqrt(1 + C_r^2)).

### 5. Single-pass crossflow, exactly one fluid mixed

C_max perfectly mixed across its flow cross-section, C_min unmixed
(`flow="crossflow_c_max_mixed"`):

```
eps = (1 / C_r) * [1 - exp(-C_r * (1 - exp(-NTU)))]
```

C_min perfectly mixed, C_max unmixed (`flow="crossflow_c_min_mixed"`):

```
eps = 1 - exp[-(1 / C_r) * (1 - exp(-NTU * C_r))]
```

Both reduce to eps = 1 - exp(-NTU) at C_r = 0 and coincide at C_r = 1.
At C_r = 0 the implementation switches to that anchor explicitly because
the literal expressions are 0/0 limit forms there.

## LMTD derivation summary (parallel and counter flow only)

Let a be the cumulative heat-transfer area measured from the hot inlet
and dT(a) = T_h(a) - T_c(a). Energy balances on a differential slice give

```
parallel flow:  d(dT)/da = -U (1/C_h + 1/C_c) dT
counter flow:   d(dT)/da = -U (1/C_h - 1/C_c) dT
```

(the cold stream flows toward decreasing a in a counter-flow
exchanger). Integrating, dT is exponential in a for both arrangements, so
with the terminal differences

```
counter flow:  dT_1 = T_h,in - T_c,out,   dT_2 = T_h,out - T_c,in
parallel flow: dT_1 = T_h,in - T_c,in,    dT_2 = T_h,out - T_c,out
```

the total duty can be written in mean-temperature-difference form,

```
Q = U A dT_lm,       dT_lm = (dT_1 - dT_2) / ln(dT_1 / dT_2)
```

with the limit dT_lm = dT_1 when dT_1 = dT_2 (which occurs at C_r = 1 in
counter flow). This is an independent route to the same duty as the
eps-NTU relations and is used as the strongest cross-check in the test
suite. With hi=max(dT_1,dT_2), lo=min(dT_1,dT_2), the code uses
log1p((hi−lo)/lo) near equality and log(hi)−log(lo) for wide ratios.
The equality limit uses lo+(hi−lo)/2, avoiding overflow in (hi+lo)/2.
Positive finite inputs spanning 1e-300 to 1e300 are covered by regressions.
This helper cannot recover information already lost when terminal
differences are formed by subtracting nearly equal outlet temperatures.

The LMTD method with F = 1 is exact only for parallel and counter flow.
For the shell-and-tube and crossflow arrangements a geometry correction
factor F(NTU-like parameters) < 1 is required; F-factor methods are out
of scope, so `terminal_lmtd_K` / `lmtd_duty_W` raise
`NotImplementedError` for those arrangements.

## Position-resolved temperature profiles (counter flow)

`s in [0, 1]` is the fractional cumulative transfer area from the hot
inlet: s = 0 is the hot-inlet / cold-outlet end, s = 1 is the
hot-outlet / cold-inlet end, A(s) = s * A_tot. The model ODEs

```
dT_h/dA = -(U/C_h)(T_h - T_c)
dT_c/dA = -(U/C_c)(T_h - T_c)
```

have the closed-form solution (with dT_0 the terminal difference at s = 0)

```
dT(s) = dT_0 * exp(-k A_tot s),   k = U (1/C_h - 1/C_c)
T_h(s) = T_h,in - [C_c / (C_c - C_h)] * (dT_0 - dT(s))
T_c(s) = T_h(s) - dT(s)
```

and, for equal capacity rates (k = 0), dT is constant at dT_0 and both
profiles are linear in s. The expression above explains the model but
is poorly conditioned when C_h > C_c at large NTU: dT_0 can round to
zero while exp(−k A_tot s) grows exponentially.

The implementation instead normalizes the integral of a decaying gap.
Let b = NTU*(C_max−C_min)/C_max >=0 and

    F(x) = expm1(−b*x) / expm1(−b), with F(x)=x at b=0.

If C_h <= C_c, use x=s and evaluate

    T_h(s) = T_h,in − (Q/C_h) F(s)
    T_c(s) = T_c,out − (Q/C_c) F(s).

If C_h > C_c, use x=1−s and evaluate backwards from the other end:

    T_h(s) = T_h,out + (Q/C_h) F(1−s)
    T_c(s) = T_c,in + (Q/C_c) F(1−s).

Both branches stay bounded and avoid an exponentially growing evaluation.
They reproduce rating endpoints to floating-point precision. Tests check
the full field against 80-digit arithmetic and independent RK4 solutions
integrated in the stable direction, including NTU 100. Temperature gaps
smaller than floating-point resolution may still display as zero; this
does not invalidate the bounded temperature profiles.

Input configuration numbers are normalized to floats after validation.
Nonfinite/zero derived conductance UA, NTU or duty raises ValueError,
including overflow or underflow of an otherwise individually finite input
combination. Rescale the physical units or operating point rather than
using an infinite/zero rating as if it were a representable solution.

## Symbols and units

| Symbol | Meaning | Units |
|---|---|---|
| U | overall heat-transfer coefficient | W/(m^2 K) |
| A | heat-transfer surface area | m^2 |
| C_h, C_c | hot / cold capacity rate | W/K |
| T_h,in, T_c,in | hot / cold inlet temperature | K |
| T_h,out, T_c,out | hot / cold outlet temperature | K |
| Q | duty (heat-transfer rate) | W |
| eps | effectiveness | dimensionless |
| NTU | number of transfer units, U A / C_min | dimensionless |
| C_r | capacity-rate ratio, C_min / C_max | dimensionless |
| dT_lm | logarithmic mean temperature difference | K |

## References by title

- Incropera et al., *Fundamentals of Heat and Mass Transfer*.
- Shah & Sekulic, *Fundamentals of Heat Exchanger Design*.
