#!/usr/bin/env python3
"""
mb_running_uncertainty.py -- error budget for m_b(2 GeV) from four-loop running.

Supports the m_b/m_d row of the predictions table in "The inverse branch of
the compact family cycle": the quoted measured ratio 1057 +/- sigma_R needs a
denominator with stated provenance. The manuscript's m_b(2 GeV) = 4.97 GeV is
a four-loop-running output; this script assigns its uncertainty.

Method: four-loop QCD beta/gamma integration (Baikov-Chetyrkin-Kuhn and
van Ritbergen-Vermaseren-Larin coefficients) from m_b(m_b) = 4186 MeV down to
2 GeV, with two-loop decoupling matching at the bottom threshold
(zeta_as = 1 + (11/72)(alpha/pi)^2, zeta_m = 1 + (89/432)(alpha/pi)^2 at
mu = m). Uncertainty components:
  (i)   alpha_s(m_Z) = 0.1180 +/- 0.0009   (dominant)
  (ii)  truncation: 3-loop vs 4-loop running shift
  (iii) matching: size of the two-loop decoupling corrections
  (iv)  input m_b(m_b) = 4186 +/- 6 MeV

Reference output (pinned):
  m_b(2 GeV) = 4952 MeV
  alpha_s(mZ) +/-0.0009 : +/-20 MeV
  truncation (3 vs 4 lp): +/-7 MeV
  matching corrections  : +/-1 MeV
  input m_b(m_b) +/-6   : +/-7 MeV
  TOTAL                 : +/-23 MeV (0.46%)

Folded into the ratio against m_d = 4.70 +/- 0.07 MeV:
  sigma_R = sqrt( (m_b/m_d^2 sigma_md)^2 + (sigma_mb/m_d)^2 ) = 16.4
  (m_d-propagated alone: 15.7; running is subleading but non-negligible)
"""
import numpy as np
from scipy.integrate import solve_ivp

Z3, Z4, Z5 = 1.202056903159594, 1.082323233711138, 1.036927755143370


def beta_coef(nf):
    return (11 - 2 * nf / 3, 102 - 38 * nf / 3,
            2857 / 2 - 5033 * nf / 18 + 325 * nf**2 / 54,
            (149753 / 6 + 3564 * Z3) - (1078361 / 162 + 6508 * Z3 / 27) * nf
            + (50065 / 162 + 6472 * Z3 / 81) * nf**2 + 1093 * nf**3 / 729)


def gam_coef(nf):
    return (4, 202 / 3 - 20 * nf / 9,
            1249 - (2216 / 27 + 160 * Z3 / 3) * nf - 140 * nf**2 / 81,
            (4603055 / 162 + 135680 * Z3 / 27 - 8800 * Z5)
            - (91723 / 27 + 34192 * Z3 / 9 - 880 * Z4 / 9 - 18400 * Z5 / 9) * nf
            + (5242 / 243 + 800 * Z3 / 9 - 160 * Z4 / 3) * nf**2
            - (332 / 243 + 64 * Z3 / 27) * nf**3)


def run_nl(mu0, mu1, a0, lnm0, nf, nloop=4):
    bb = list(beta_coef(nf)[:nloop]) + [0.0] * (4 - nloop)
    gg = list(gam_coef(nf)[:nloop]) + [0.0] * (4 - nloop)

    def rhs(t, y):
        a, lnm = y
        return [-a * a * (bb[0] + bb[1] * a + bb[2] * a * a + bb[3] * a**3),
                -a * (gg[0] + gg[1] * a + gg[2] * a * a + gg[3] * a**3)]

    sol = solve_ivp(rhs, (2 * np.log(mu0), 2 * np.log(mu1)), [a0, lnm0],
                    rtol=1e-11, atol=1e-14)
    return sol.y[0, -1], sol.y[1, -1]


def decouple(a, lnm, match=True):
    if not match:
        return a, lnm
    ap = a * 4.0                      # alpha/(4 pi) -> alpha/pi
    return a * (1 + (11 / 72) * ap**2), lnm + np.log(1 + (89 / 432) * ap**2)


def mb_2gev(aZ=0.1180, mb_mb=4186.0, nloop=4, match=True):
    a_mt, _ = run_nl(91.1876, 162.5, aZ / (4 * np.pi), 0.0, 5, nloop)
    a_mb, lnm_mb = run_nl(162.5, 4.186, a_mt, np.log(162.5e3), 5, nloop)
    a_mb, lnm_mb = decouple(a_mb, lnm_mb, match)
    _, lnb_2 = run_nl(4.186, 2.0, a_mb, np.log(mb_mb), 4, nloop)
    return np.exp(lnb_2)


def budget():
    base = mb_2gev()
    d_as = abs(mb_2gev(0.1189) - mb_2gev(0.1171)) / 2
    d_tr = abs(mb_2gev(nloop=3) - base)
    d_ma = abs(mb_2gev(match=False) - base)
    d_in = abs(mb_2gev(mb_mb=4186 + 6) - mb_2gev(mb_mb=4186 - 6)) / 2
    tot = np.sqrt(d_as**2 + d_tr**2 + d_ma**2 + d_in**2)
    print(f"m_b(2 GeV) = {base:.0f} MeV")
    print(f"  alpha_s(mZ) +/-0.0009 : +/-{d_as:.0f} MeV")
    print(f"  truncation (3 vs 4 lp): +/-{d_tr:.0f} MeV")
    print(f"  matching corrections  : +/-{d_ma:.0f} MeV")
    print(f"  input m_b(m_b) +/-6   : +/-{d_in:.0f} MeV")
    print(f"  TOTAL                 : +/-{tot:.0f} MeV ({tot/base:.2%})")
    sig_md = base * 0.07 / 4.70**2
    sig_R = np.sqrt(sig_md**2 + (tot / 4.70)**2)
    print(f"ratio error budget: sigma_md-alone = {sig_md:.1f}, "
          f"sigma_R = {sig_R:.1f}")
    return base, tot, sig_R


if __name__ == "__main__":
    budget()
