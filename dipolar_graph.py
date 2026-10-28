"""
dipolar_graph.py: coupling matrix K_ij and connectivity graphs for 2D STNO arrays.

Uses the same dipolar kernel as stno_dipolar_v3_2.py. For two oscillators with precession
axes n_i, n_j and bond direction e (c = n.e), the co-rotating coupling is

    K_ij = d_ij * (3 c_i c_j - n_i.n_j) / 2,    d_ij = V / (4 pi r_ij^3)

Small-cone approximation only. Units are d(r_ref) unless V is passed.
"""
import numpy as np
from scipy.sparse.csgraph import connected_components, shortest_path


# geometry

def lattice_positions(nx, ny, r, jitter=0.0, seed=0):
    # node index = row*nx + col, all in the x-y plane
    xs, ys = np.meshgrid(np.arange(nx) * r, np.arange(ny) * r)
    pos = np.stack([xs.ravel(), ys.ravel(), np.zeros(nx * ny)], axis=1)
    if jitter > 0:
        pos[:, :2] += np.random.default_rng(seed).normal(0, jitter * r, (nx * ny, 2))
    return pos


def uniform_axes(N, tilt_deg=0.0, azimuth_deg=0.0):
    th, ph = np.radians(tilt_deg), np.radians(azimuth_deg)
    n = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    return np.tile(n, (N, 1))


def checker_axes(nx, ny, tilt_deg):
    # tilt azimuth alternates 0/180 deg like a checkerboard
    ax = np.zeros((nx * ny, 3))
    th = np.radians(tilt_deg)
    for row in range(ny):
        for col in range(nx):
            s = 1.0 if (row + col) % 2 == 0 else -1.0
            ax[row * nx + col] = [s * np.sin(th), 0.0, np.cos(th)]
    return ax


# coupling matrix and graphs

def dipolar_d(r, V=1.0):
    return V / (4 * np.pi * r ** 3)


def coupling_matrix(pos, axes, V=1.0, disk_R=None, normalize=True):
    # symmetric, zero diagonal. disk_R adds the finite-size factor (1 + (R/r)^2)^-1.5
    # that dipole_factors() in the simulator uses
    N = len(pos)
    K = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            rv = pos[i] - pos[j]
            r = np.linalg.norm(rv)
            e = rv / r
            ci, cj = axes[i] @ e, axes[j] @ e
            k = dipolar_d(r, V) * (3 * ci * cj - axes[i] @ axes[j]) / 2
            if disk_R is not None:
                k *= (1 + (disk_R / r) ** 2) ** -1.5
            K[i, j] = K[j, i] = k
    if normalize and np.abs(K).max() > 0:
        K = K / np.abs(K).max()
    return K


def dipolar_mask(K, rel_thresh=0.05):
    # keep links with |K| above rel_thresh * max|K|
    M = np.abs(K) > rel_thresh * np.abs(K).max()
    np.fill_diagonal(M, False)
    return M


def distance_mask(pos, cutoff):
    D = np.linalg.norm(pos[:, None] - pos[None], axis=2)
    M = (D <= cutoff * (1 + 1e-9))
    np.fill_diagonal(M, False)
    return M


def all_to_all_mask(N):
    M = np.ones((N, N), bool)
    np.fill_diagonal(M, False)
    return M


def graph_stats(mask, K=None):
    N = len(mask)
    n_comp, _ = connected_components(mask, directed=False)
    deg = mask.sum(1)
    sp = shortest_path(mask.astype(float), directed=False, unweighted=True)
    finite = sp[np.isfinite(sp)]
    st = dict(N=N, edges=int(mask.sum() // 2), mean_degree=float(deg.mean()),
              min_degree=int(deg.min()), components=int(n_comp),
              diameter=float(finite.max()) if n_comp == 1 else np.inf)
    if K is not None:
        up = np.triu(mask, 1)
        vals = K[up]
        st['frac_positive'] = float((vals > 0).mean()) if len(vals) else np.nan
        st['frac_negative'] = float((vals < 0).mean()) if len(vals) else np.nan
        st['strength_ratio_max_min'] = float(np.abs(vals).max() / np.abs(vals).min()) if len(vals) else np.nan
    return st


def default_roles(nx, ny, n_in=2, n_out=1):
    # reference + inputs on the left column, outputs on the right
    idx = lambda row, col: row * nx + col
    rows_in = np.linspace(0, ny - 1, n_in + 1).round().astype(int)[1:]
    ref = idx(0, 0)
    inputs = [idx(r, 0) for r in rows_in if idx(r, 0) != ref][:n_in]
    rows_out = np.linspace(0, ny - 1, n_out + 2).round().astype(int)[1:-1]
    outputs = [idx(r, nx - 1) for r in rows_out]
    return dict(ref=ref, inputs=inputs, outputs=outputs)


def graph_figure(sizes=((3, 3), (4, 4), (5, 5)), r=80e-9, thresh=0.05, tilt_deg=45.0,
                 out='dipolar_graphs'):
    """K_ij heatmaps, |K_ij| vs distance, and a degree/diameter table for square lattices
    (N = 9, 16, 25), compared with 4NSQ and all-to-all. K in units of d(r).
    Writes <out>.png, .pdf and _table.csv."""
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.ticker import MaxNLocator, ScalarFormatter, NullFormatter

    d0 = dipolar_d(r, 1.0)
    fig = plt.figure(figsize=(10.5, 14))
    gs = fig.add_gridspec(4, len(sizes), height_ratios=[1, 1, 1, 0.9], hspace=0.5, wspace=0.75)
    rows = []
    for c, (nx, ny) in enumerate(sizes):
        N = nx * ny
        pos = lattice_positions(nx, ny, r)
        cfg = {'axis z': uniform_axes(N, 0.0), f'axis tilted {tilt_deg:.0f} deg': uniform_axes(N, tilt_deg)}
        Ks = {k: coupling_matrix(pos, ax, normalize=False) / d0 for k, ax in cfg.items()}
        D = np.linalg.norm(pos[:, None] - pos[None], axis=2) / r
        iu = np.triu_indices(N, 1)
        for rr, (name, K) in enumerate(Ks.items()):
            ax = fig.add_subplot(gs[rr, c])
            lim = np.abs(K).max()
            im = ax.imshow(K, cmap='RdBu_r', norm=TwoSlopeNorm(0.0, -lim, lim))
            ax.set_title(f'N = {N}, {name}', fontsize=8)
            ax.set_xlabel('oscillator j')
            ax.set_ylabel('oscillator i')
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            cb.set_label(r'$K_{ij}$ / $d(r)$', fontsize=8)
        ax = fig.add_subplot(gs[2, c])
        for (name, K), mk in zip(Ks.items(), ['o', 's']):
            ax.scatter(D[iu], np.abs(K[iu]), s=10, marker=mk, alpha=0.6, label=name)
        rg = np.linspace(1, D.max(), 100)
        ax.plot(rg, 0.5 * rg ** -3, 'k--', lw=0.8, label=r'$0.5\,(r_{ij}/r)^{-3}$')
        ax.axhline(thresh * max(np.abs(K).max() for K in Ks.values()), color='gray', ls=':', lw=0.8)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_ylim(bottom=max(1e-3, ax.get_ylim()[0]))
        ax.set_xticks([1, 2, 3, 4, 6])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel('distance r_ij / r')
        ax.set_ylabel(r'$|K_{ij}|$ / $d(r)$')
        ax.legend(fontsize=6)
        masks = {'NN square': distance_mask(pos, r * 1.01), '4NSQ (<=2r)': distance_mask(pos, r * 2.01)}
        for name, K in Ks.items():
            masks[f'dipolar {name} (>{thresh * 100:.0f}%)'] = dipolar_mask(K, thresh)
        masks['all-to-all'] = all_to_all_mask(N)
        for name, m in masks.items():
            s = graph_stats(m)
            rows.append([N, name, s['edges'], round(s['mean_degree'], 2), s['min_degree'],
                         s['components'], s['diameter']])
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')
    cols = ['N', 'graph', 'edges', 'mean degree', 'min degree', 'components', 'diameter (hops)']
    cell = [[str(x) if not (isinstance(x, float) and np.isinf(x)) else 'inf' for x in row]
            for row in rows]
    tb = ax.table(cellText=cell, colLabels=cols, loc='center', cellLoc='center')
    tb.auto_set_font_size(False)
    tb.set_fontsize(6.5)
    tb.auto_set_column_width(col=list(range(len(cols))))
    fig.suptitle(f'Dipolar coupling graphs, r = {r * 1e9:.0f} nm, threshold {thresh * 100:.0f} % of max |K|',
                 fontsize=9, y=0.995)
    fig.savefig(out + '.png', dpi=170, bbox_inches='tight')
    fig.savefig(out + '.pdf', bbox_inches='tight')
    with open(out + '_table.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(cols)
        w.writerows(rows)
    print(f"saved {out}.png / .pdf / _table.csv")
    for row in rows:
        print(row)


if __name__ == "__main__" and '--figure' in __import__('sys').argv:
    graph_figure()
elif __name__ == "__main__":
    nx = ny = 4
    pos = lattice_positions(nx, ny, 80e-9)
    for name, ax in [('axis z', uniform_axes(16, 0)), ('tilt 45', uniform_axes(16, 45)),
                     ('checker 45', checker_axes(nx, ny, 45))]:
        K = coupling_matrix(pos, ax)
        for th in (0.05, 0.2):
            print(name, f"thr={th}", graph_stats(dipolar_mask(K, th), K))
