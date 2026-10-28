"""
check_K.py: compare the analytic dipolar K_ij (dipolar_graph.py) with the effective coupling
from the macrospin simulator (stno_dipolar_v3_2.py).

Setup: two identical STNOs (Chen type, k = 0), bond angle psi from the precession axis z,
r = 120 nm, co-rotating pair, T = 0. lock_range() bisects on the detuning (current of
oscillator 2) and returns the one-sided locking range in Hz (same routine as test H2).
K_eff comes from inverting the Chen-2014 estimate used in the simulator:

    range = f_M * 2|K| / alpha   ->   K_eff = alpha * range / (2 f_M)

That relation is only order-of-magnitude (nonlinearity nu and finite amplitude change the
prefactor), so K_eff/|K| need not be 1. The more useful checks are the scaling with psi
(range/range_ref vs |K|/|K|_ref) and the sign of K (locked phase at zero detuning).

usage: MPLBACKEND=Agg python check_K.py
       python check_K.py --angles 0 30 90 --r 120      (deg, nm)
Writes check_K.csv. Single run, no seed averaging, T = 0.
"""
import sys
import csv
import numpy as np
import stno_dipolar_v3_2 as S
from dipolar_graph import coupling_matrix, uniform_axes


def main(angles=(0.0, 30.0, 90.0), r_nm=120.0, iters=8, out='check_K.csv'):
    p = S.default_params(k=0.0)
    r = r_nm * 1e-9
    V = S.xsec(p) * p['d']
    d = V / (4 * np.pi * r ** 3)
    fM = S.f_M(p)
    rows = []
    for psi in angles:
        ps = np.radians(psi)
        pos = np.array([[0, 0, 0], [r * np.sin(ps), 0, r * np.cos(ps)]])
        K_an = coupling_matrix(pos, uniform_axes(2, 0.0), V=V, normalize=False)[0, 1]
        K_formula = S.coupling_K(d, psi)
        assert abs(K_an - K_formula) < 1e-12 * max(1.0, abs(d)), (K_an, K_formula)
        rng_sim, note = S.lock_range(psi, 'co', p, r, 1e-3, iters=iters)
        rng_pred = S.chen_lock_estimate(d, psi, p)
        m0 = S.pair_run(psi, kval=0.0, r=r, T=60e-9)  # locked phase at zero detuning
        rows.append(dict(psi=psi, d=d, K_an=K_an, range_pred=rng_pred, range_sim=rng_sim,
                         note=note, K_eff=p['alpha'] * rng_sim / (2 * fM),
                         phase_class=S.classify(m0), cos=m0['cos']))
        print(f"psi={psi:5.1f}: K_an={K_an:+.3e}  sim range={rng_sim / 1e6:8.1f} MHz ({note})  "
              f"Chen estimate={rng_pred / 1e6:8.1f} MHz  zero-detuning: {m0['cos']:+.2f} "
              f"{S.classify(m0)}", flush=True)
    ref = max(rows, key=lambda x: abs(x['K_an']))
    for x in rows:
        x['ratio_abs'] = x['K_eff'] / abs(x['K_an']) if x['K_an'] != 0 else np.nan
        x['rel_K'] = abs(x['K_an']) / abs(ref['K_an'])
        x['rel_range'] = x['range_sim'] / ref['range_sim'] if ref['range_sim'] > 0 else np.nan
    hdr = ['psi_deg', 'K_analytic', 'range_Chen_MHz', 'range_sim_MHz', 'sim_note', 'K_eff_from_sim',
           'K_eff/|K_an|', '|K|/|K|ref', 'range/range_ref', 'zero_detuning_cos', 'phase_state']
    table = [[x['psi'], f"{x['K_an']:+.3e}", f"{x['range_pred'] / 1e6:.1f}", f"{x['range_sim'] / 1e6:.1f}",
              x['note'], f"{x['K_eff']:.3e}", f"{x['ratio_abs']:.2f}", f"{x['rel_K']:.2f}",
              f"{x['rel_range']:.2f}", f"{x['cos']:+.2f}", x['phase_class']] for x in rows]
    with open(out, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(table)
    print(f"\nr = {r_nm:.0f} nm, d = {d:.3e}, f_M = {fM / 1e9:.2f} GHz, alpha = {p['alpha']}, "
          f"k = 0, T = 0, bisection iters = {iters}, reference psi = {ref['psi']}")
    print(' | '.join(hdr))
    for t in table:
        print(' | '.join(str(c) for c in t))
    print(f"saved {out}")


if __name__ == "__main__":
    a = sys.argv
    angles = [float(x) for x in a[a.index('--angles') + 1:a.index('--angles') + 4]] if '--angles' in a else (0.0, 30.0, 90.0)
    r_nm = float(a[a.index('--r') + 1]) if '--r' in a else 120.0
    main(angles, r_nm)
