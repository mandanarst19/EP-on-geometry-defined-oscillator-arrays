"""
ep_xy.py: equilibrium propagation on an XY (Kuramoto) phase network with a fixed mask.
Baseline for Wang, Wanjura, Marquardt.

E(theta) = -sum_{i<j} J_ij cos(theta_i - theta_j),  J = W * mask
C(theta) = sum_o [1 - cos(theta_o - phi_o)]
dtheta/dt = -dF/dtheta (+ noise),  F = E + beta*C, free nodes only
EP gradient: dL/dJ_ij ~ (1/2beta) [-cos(d_ij)|_{+beta} + cos(d_ij)|_{-beta}]

Clamped: one reference (phase 0) and the inputs. States are batched, theta is (B, N).
"""
import numpy as np


class XYNet:
    def __init__(self, mask, ref, inputs, outputs, seed=0, w_scale=0.5):
        self.mask = np.triu(np.asarray(mask, bool), 1)
        self.mask = self.mask | self.mask.T
        self.N = len(mask)
        self.ref, self.inputs, self.outputs = ref, list(inputs), list(outputs)
        self.clamped = np.array([ref] + self.inputs)
        self.free = np.setdiff1d(np.arange(self.N), self.clamped)
        self.rng = np.random.default_rng(seed)
        W = self.rng.normal(0, w_scale, (self.N, self.N))
        W = np.triu(W, 1)
        self.W = (W + W.T) * self.mask

    # physics
    def grad_E(self, th):
        d = th[:, :, None] - th[:, None, :]
        return (self.W[None] * np.sin(d)).sum(2)

    def relax(self, th0, x_phase, target=None, beta=0.0, steps=400, dt=0.1, tol=1e-6,
              temp=0.0, rng=None):
        # gradient descent on the free nodes; x_phase is (B, n_in)
        th = th0.copy()
        th[:, self.ref] = 0.0
        th[:, self.inputs] = x_phase
        fr = self.free
        for _ in range(steps):
            g = self.grad_E(th)
            if beta != 0.0 and target is not None:
                g = g.copy()
                g[:, self.outputs] += beta * np.sin(th[:, self.outputs] - target)
            step = dt * g[:, fr]
            th[:, fr] -= step
            if temp > 0:
                th[:, fr] += np.sqrt(2 * temp * dt) * (rng or self.rng).normal(size=step.shape)
            elif np.abs(step).max() < tol * dt:
                break
        return th

    def init_state(self, B):
        return self.rng.uniform(-np.pi, np.pi, (B, self.N))

    # gradients
    def cost(self, th, target):
        return (1 - np.cos(th[:, self.outputs] - target)).sum(1)

    def ep_grad(self, x_phase, target, beta=0.05, th_free=None, symmetric=True, **kw):
        # returns (dW, th_free); dW is the batch mean on the mask
        B = len(x_phase)
        if th_free is None:
            th_free = self.relax(self.init_state(B), x_phase, **kw)
        if symmetric:
            thp = self.relax(th_free, x_phase, target, +beta, **kw)
            thm = self.relax(th_free, x_phase, target, -beta, **kw)
            cp = np.cos(thp[:, :, None] - thp[:, None, :])
            cm = np.cos(thm[:, :, None] - thm[:, None, :])
            dW = (-cp + cm).mean(0) / (2 * beta)
        else:
            thn = self.relax(th_free, x_phase, target, beta, **kw)
            c0 = np.cos(th_free[:, :, None] - th_free[:, None, :])
            cn = np.cos(thn[:, :, None] - thn[:, None, :])
            dW = (-cn + c0).mean(0) / beta
        return dW * self.mask, th_free

    def fd_grad(self, x_phase, target, th0, eps=1e-4, **kw):
        # finite differences of the free-state loss, every relaxation starts from th0
        # (needs a tight tol)
        G = np.zeros((self.N, self.N))
        iu = np.argwhere(np.triu(self.mask, 1))
        for i, j in iu:
            L = []
            for s in (+eps, -eps):
                W0 = self.W.copy()
                self.W[i, j] += s
                self.W[j, i] += s
                th = self.relax(th0, x_phase, **kw)
                L.append(self.cost(th, target).mean())
                self.W = W0
            G[i, j] = G[j, i] = (L[0] - L[1]) / (2 * eps)
        return G

    # inference / training
    def predict(self, x_phase, n_init=4, **kw):
        # several random inits to catch multistability
        B = len(x_phase)
        outs = []
        for _ in range(n_init):
            th = self.relax(self.init_state(B), x_phase, **kw)
            outs.append(th[:, self.outputs])
        return np.array(outs)  # (n_init, B, n_out)

    def accuracy(self, x_phase, target, n_init=4, **kw):
        outs = self.predict(x_phase, n_init, **kw)
        ok = np.cos(outs - target[None]) > 0
        return float(ok.all(2).mean())


def adam_update(W, g, state, lr=0.02, b1=0.9, b2=0.999, eps=1e-8):
    state['t'] = state.get('t', 0) + 1
    state['m'] = b1 * state.get('m', 0) + (1 - b1) * g
    state['v'] = b2 * state.get('v', 0) + (1 - b2) * g * g
    mh = state['m'] / (1 - b1 ** state['t'])
    vh = state['v'] / (1 - b2 ** state['t'])
    return W - lr * mh / (np.sqrt(vh) + eps)


def train(net, X, Y, epochs=1500, lr=0.02, beta=0.05, n_init=4, eval_every=50, w_clip=5.0,
          verbose=False, **kw):
    # full-batch EP. X: (B, n_in) input phases, Y: (B, n_out) target phases.
    # "converged" = accuracy 1.0 on all samples for all n_init inits
    st, hist = {}, dict(epoch=[], acc=[], loss=[])
    converged_at = None
    for ep in range(epochs + 1):
        if ep % eval_every == 0:
            th = net.relax(net.init_state(len(X)), X, **kw)
            loss = float(net.cost(th, Y).mean())
            acc = net.accuracy(X, Y, n_init, **kw)
            hist['epoch'].append(ep), hist['acc'].append(acc), hist['loss'].append(loss)
            if verbose:
                print(f"  ep {ep:5d} loss {loss:.4f} acc {acc:.2f}")
            if acc == 1.0 and converged_at is None:
                converged_at = ep
                break
        if ep == epochs:
            break
        dW, _ = net.ep_grad(X, Y, beta=beta, **kw)
        # Adam on the full matrix, then keep upper triangle and symmetrise
        Wn = adam_update(net.W, dW, st, lr=lr)
        Wn = np.triu(Wn, 1)
        net.W = np.clip(Wn + Wn.T, -w_clip, w_clip) * net.mask
    hist['converged_at'] = converged_at
    return hist


def xor_data():
    # inputs are phases {0, pi}, target phase is {0, pi} = XOR of the bits
    bits = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    X = bits * np.pi
    Y = ((bits[:, 0] ^ bits[:, 1]) * np.pi)[:, None]
    return X.astype(float), Y.astype(float)
