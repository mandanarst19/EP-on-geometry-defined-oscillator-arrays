# stno-ep-dipolar

Equilibrium propagation (EP) on arrays of spin-torque nano-oscillators (STNOs) whose
connectivity is set by dipolar coupling geometry. Early-stage code: the training runs on the
reduced XY (Kuramoto) model, and the macrospin simulator is used only to check the coupling.

## Layout

| File | What it does |
|---|---|
| `dipolar_graph.py` | positions + precession axes -> coupling matrix K_ij (small-cone limit), connectivity masks, graph statistics, summary figure |
| `ep_xy.py` | EP for an XY phase network with a fixed mask (free/nudged phases, symmetric nudge, Adam, multi-init readout) |
| `test_gradient.py` | cosine similarity of the EP gradient vs finite differences, beta sweep |
| `run_xor.py` | XOR convergence for all-to-all, NN, 4NSQ and dipolar graphs |
| `check_K.py` | analytic K vs locking range from the macrospin simulator |
| `results/` | CSV and PDF outputs from the runs below |

`check_K.py` imports `stno_dipolar_v3_2.py` (the macrospin LLGS simulator), which is **not**
included in this repository. Put it next to the scripts to run that check. The other scripts
do not need it.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python dipolar_graph.py --figure            # graph figure + table (dipolar_graphs.*)
python test_gradient.py                     # quick EP vs finite-difference check (~1 min)
python test_gradient.py --sweep --seeds 5   # beta sweep -> ep_gradient_vs_beta.*
python test_gradient.py --replot            # redraw from the saved csv
python run_xor.py 5 1500                    # XOR training, 5 seeds, 1500 epochs (~1 h)
MPLBACKEND=Agg python check_K.py            # needs stno_dipolar_v3_2.py
```

Scripts write their outputs to the current directory.

## Results so far

**Coupling graphs** (`results/dipolar_graphs.pdf`, `results/dipolar_graphs_table.csv`):
square lattices with N = 9, 16, 25 at r = 80 nm. With the axis along z, the 5 % dipolar graph
has mean degree 7.6 / 10.3 / 12.0 and diameter 2 / 2 / 3, between 4NSQ and all-to-all.

**EP gradient quality** (`results/ep_gradient_vs_beta.*`): N = 16, XOR batch, 5 seeds.
The cosine similarity with the finite-difference gradient is about 1 for beta <= 0.1 on all
three graphs. At beta = 0.3 and 1 it drops, and the dipolar graph degrades most and varies
most between seeds.

**K check against the simulator** (`results/check_K.csv`): pair of identical STNOs, r = 120 nm.

| psi (deg) | K analytic | sim range (MHz) | range / range_ref | \|K\| / \|K\|_ref | zero-detuning phase |
|---|---|---|---|---|---|
| 0  | +4.56e-4 | 1163.8 | 1.00 | 1.00 | anti-phase |
| 30 | +2.85e-4 | 730.4  | 0.63 | 0.62 | anti-phase |
| 90 | -2.28e-4 | 569.5  | 0.49 | 0.50 | in-phase |

The locking range scales with |K| as predicted, and the sign of K matches the locked phase.
The absolute ratio K_eff / |K| is about 0.84, which is reasonable since the inversion used
is only an order-of-magnitude estimate. Single run, T = 0, no seed averaging.

## Limitations

- The simulator is not in the training loop yet.
- Every masked coupling is trainable with either sign. In a real array the geometry fixes the
  sign and size of K_ij, so W has to be replaced by physical knobs (currents, detunings,
  positions, tilts).
- The XOR encoding (inputs 0/pi, target 0/pi, one reference node) is a first choice, not the
  exact protocol of Wang et al.
- XOR training results are not included yet; the runs so far (3 seeds, 600 epochs) only show trends.

## TODO

- [ ] verify K_ij on a 2x2 array
- [ ] replace W by physical parameters, fixed-sign K
- [ ] amplitude-phase model (Rageau-Grollier)
- [ ] BPTT reference gradient (JAX/PyTorch), noise statistics
- [ ] Iris / MNIST

## License

MIT, see `LICENSE`.
