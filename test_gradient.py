"""EP gradient vs finite-difference gradient of the free-state loss.
usage: python test_gradient.py              quick check
       python test_gradient.py --sweep      beta sweep (--seeds N)
       python test_gradient.py --replot     redraw figure from the saved csv
BPTT on the simulated dynamics is the final reference; this is the reduced-model version."""
import numpy as np
from ep_xy import XYNet, xor_data
from dipolar_graph import lattice_positions, uniform_axes, coupling_matrix, dipolar_mask, \
    distance_mask, all_to_all_mask, default_roles


def cosine(a, b, mask):
    iu = np.triu(mask, 1)
    x, y = a[iu], b[iu]
    return float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y) + 1e-30))


def check(name, mask, nx, ny, betas=(0.2, 0.05, 0.01), seed=0):
    roles = default_roles(nx, ny)
    net = XYNet(mask, roles['ref'], roles['inputs'], roles['outputs'], seed=seed)
    X, Y = xor_data()
    kw = dict(steps=4000, dt=0.1, tol=1e-10)
    th0 = net.init_state(len(X))
    thf = net.relax(th0, X, **kw)
    G = net.fd_grad(X, Y, th0, **kw)
    row = []
    for b in betas:
        dW, _ = net.ep_grad(X, Y, beta=b, th_free=thf, **kw)
        row.append(cosine(dW, G, net.mask))
    print(f"{name:22s} edges={int(net.mask.sum() // 2):3d}  cos(EP, FD) for beta {betas}: "
          + "  ".join(f"{c:.4f}" for c in row))


def beta_sweep(betas=None, n_seeds=5, nx=4, ny=4, r=80e-9, out='ep_gradient_vs_beta'):
    # cos(EP, FD) vs beta. A seed changes the initial W and phases; the FD gradient is
    # computed once per (graph, seed) and reused for all betas.
    import csv
    betas = np.logspace(-3, 0, 7) if betas is None else np.asarray(betas, float)
    N = nx * ny
    pos = lattice_positions(nx, ny, r)
    K = coupling_matrix(pos, uniform_axes(N, 0))
    graphs = {'all-to-all': all_to_all_mask(N),
              '4NSQ (<= 2 r)': distance_mask(pos, r * 2.01),
              'dipolar (axis z, >5 %)': dipolar_mask(K, 0.05)}
    roles = default_roles(nx, ny)
    X, Y = xor_data()
    kw = dict(steps=4000, dt=0.1, tol=1e-10)
    res = {g: np.zeros((n_seeds, len(betas))) for g in graphs}
    for g, mask in graphs.items():
        for sd in range(n_seeds):
            net = XYNet(mask, roles['ref'], roles['inputs'], roles['outputs'], seed=sd)
            th0 = net.init_state(len(X))
            thf = net.relax(th0, X, **kw)
            G = net.fd_grad(X, Y, th0, **kw)
            for k, b in enumerate(betas):
                dW, _ = net.ep_grad(X, Y, beta=b, th_free=thf, **kw)
                res[g][sd, k] = cosine(dW, G, net.mask)
            print(f"{g:24s} seed {sd}: " + " ".join(f"{c:.4f}" for c in res[g][sd]), flush=True)
    with open(out + '.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['graph', 'seed', 'beta', 'cos_EP_FD', 'n_edges', 'N'])
        for g, mask in graphs.items():
            ne = int(np.triu(mask, 1).sum())
            for sd in range(n_seeds):
                for k, b in enumerate(betas):
                    w.writerow([g, sd, f"{b:.6g}", f"{res[g][sd, k]:.8f}", ne, N])
    plot_sweep(res, betas, n_seeds, N, out)


def replot_sweep(csv_path='ep_gradient_vs_beta.csv', out='ep_gradient_vs_beta'):
    import csv
    rows = list(csv.DictReader(open(csv_path)))
    betas = np.array(sorted({float(r['beta']) for r in rows}))
    graphs = list(dict.fromkeys(r['graph'] for r in rows))
    n_seeds = len({int(r['seed']) for r in rows})
    res = {g: np.zeros((n_seeds, len(betas))) for g in graphs}
    for r in rows:
        res[r['graph']][int(r['seed']), int(np.argmin(np.abs(betas - float(r['beta']))))] = float(r['cos_EP_FD'])
    plot_sweep(res, betas, n_seeds, int(rows[0]['N']), out)


def plot_sweep(res, betas, n_seeds, N, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    for g in res:
        m = res[g].mean(0)
        s = res[g].std(0, ddof=1) if n_seeds > 1 else np.zeros(len(betas))
        # cosine can't exceed 1, so clip the upper error bar
        ax.errorbar(betas, m, yerr=[s, np.minimum(s, np.maximum(1.0 - m, 0.0))], marker='o', ms=4,
                    capsize=3, label=g)
    ax.set_xscale('log')
    ax.set_xlabel(r'nudging strength $\beta$')
    ax.set_ylabel('cos. similarity, EP vs finite-diff. gradient')
    ax.set_title(f'EP gradient check, XY model, N = {N} (XOR batch)', fontsize=9)
    ax.legend(fontsize=7, loc='lower left')
    ax.text(0.99, 0.02, f'{n_seeds} seeds, mean $\\pm$ s.d., symmetric nudge',
            transform=ax.transAxes, ha='right', va='bottom', fontsize=6.5, color='gray')
    fig.tight_layout()
    fig.savefig(out + '.png', dpi=200)
    fig.savefig(out + '.pdf')
    print(f"saved {out}.csv / .png / .pdf")


if __name__ == "__main__":
    import sys
    if '--replot' in sys.argv:
        replot_sweep()
        sys.exit(0)
    if '--sweep' in sys.argv:
        ns = int(sys.argv[sys.argv.index('--seeds') + 1]) if '--seeds' in sys.argv else 5
        beta_sweep(n_seeds=ns)
        sys.exit(0)
    nx = ny = 4
    N = nx * ny
    pos = lattice_positions(nx, ny, 80e-9)
    K = coupling_matrix(pos, uniform_axes(N, 0))
    check("all-to-all", all_to_all_mask(N), nx, ny)
    check("nearest-neighbour", distance_mask(pos, 80e-9 * 1.01), nx, ny)
    check("dipolar (thr 0.05)", dipolar_mask(K, 0.05), nx, ny)
