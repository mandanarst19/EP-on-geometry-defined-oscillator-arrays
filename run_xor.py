"""XOR with EP on different coupling graphs (reduced XY model).
Reports convergence probability over seeds and epochs to convergence.
usage: python run_xor.py [n_seeds] [epochs]"""
import sys, time, json
import numpy as np
from ep_xy import XYNet, train, xor_data
from dipolar_graph import (lattice_positions, uniform_axes, coupling_matrix,
                           dipolar_mask, distance_mask, all_to_all_mask, default_roles)


def graphs(nx, ny, r=80e-9):
    N = nx * ny
    pos = lattice_positions(nx, ny, r)
    Kz = coupling_matrix(pos, uniform_axes(N, 0))
    Kt = coupling_matrix(pos, uniform_axes(N, 45))
    return {
        'all-to-all': all_to_all_mask(N),
        'NN square': distance_mask(pos, r * 1.01),
        '4NSQ (<=2r)': distance_mask(pos, r * 2.01),
        'dipolar, axis z (>5%)': dipolar_mask(Kz, 0.05),
        'dipolar, axis z (>20%)': dipolar_mask(Kz, 0.20),
        'dipolar, tilt 45 (>5%)': dipolar_mask(Kt, 0.05),
    }


if __name__ == "__main__":
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    epochs = int(sys.argv[2]) if len(sys.argv) > 2 else 1500
    nx = ny = 4
    roles = default_roles(nx, ny)
    X, Y = xor_data()
    results = {}
    for name, mask in graphs(nx, ny).items():
        t0 = time.time()
        conv = []
        for sd in range(n_seeds):
            net = XYNet(mask, roles['ref'], roles['inputs'], roles['outputs'], seed=sd)
            h = train(net, X, Y, epochs=epochs, lr=0.02, beta=0.05, n_init=4, eval_every=25,
                      steps=300, dt=0.1)
            conv.append(h['converged_at'])
        ok = [c for c in conv if c is not None]
        results[name] = dict(edges=int(np.triu(mask, 1).sum()), conv_prob=len(ok) / n_seeds,
                             epochs=conv)
        print(f"{name:24s} edges={results[name]['edges']:3d}  P(conv)={len(ok)}/{n_seeds}  "
              f"median epochs={np.median(ok) if ok else '-'}  ({time.time() - t0:.0f} s)", flush=True)
    json.dump(results, open('xor_results.json', 'w'), indent=1)
