"""High-precision reference and explicitly obsolete forward evaluation.

The legacy function is retained ONLY to reproduce the numerical failure in
the comparison figure. It is never used by the production package.
"""
from decimal import Decimal, localcontext
import numpy as np

def decimal_profiles(config, stations):
    """80-digit direct analytic solution, independent of the stable float path.

    Deliberately uses the original forward exponential formula; 80-digit
    arithmetic resolves the small boundary difference even at NTU 100.
    """
    with localcontext() as context:
        context.prec = 80
        D = Decimal.from_float
        ch, cc = D(config.C_h_W_K), D(config.C_c_W_K)
        ua = D(config.U_W_m2K) * D(config.A_m2)
        thin, tcin = D(config.T_h_in_K), D(config.T_c_in_K)
        ntu, cr = ua / min(ch, cc), min(ch, cc) / max(ch, cc)
        if cr == 1:
            eps = ntu / (1 + ntu)
        else:
            e = (-ntu * (1-cr)).exp()
            eps = (1-e) / (1-cr*e)
        q = eps * min(ch, cc) * (thin-tcin)
        gap0 = thin - (tcin + q/cc)
        k_a = ua * (1/ch - 1/cc)
        hot, cold = [], []
        for station in stations:
            x = D(float(station))
            gap = gap0 * (-k_a*x).exp()
            th = thin - ua/ch*gap0*x if ch == cc else thin - cc/(cc-ch)*(gap0-gap)
            hot.append(float(th))
            cold.append(float(th-gap))
        return np.array(hot), np.array(cold)


def legacy_forward_profiles(config, stations):
    """Reproduce the previous float64 path, including its equality thresholds."""
    ch, cc = config.C_h_W_K, config.C_c_W_K
    ua = config.U_W_m2K * config.A_m2
    ntu, cr = ua/min(ch, cc), min(ch, cc)/max(ch, cc)
    if abs(cr-1) <= 1e-9:
        eps = ntu/(1+ntu)
    else:
        eps = -np.expm1(-ntu*(1-cr))/(1-cr*np.exp(-ntu*(1-cr)))
    duty = eps*min(ch, cc)*(config.T_h_in_K-config.T_c_in_K)
    gap0 = config.T_h_in_K-(config.T_c_in_K+duty/cc)
    if abs(ch-cc) <= 1e-12*max(ch, cc):
        gap = np.full_like(stations, gap0)
        hot = config.T_h_in_K-ua*gap0/ch*stations
    else:
        gap = gap0*np.exp(-ua*(1/ch-1/cc)*stations)
        hot = config.T_h_in_K-cc/(cc-ch)*(gap0-gap)
    return hot, hot-gap
