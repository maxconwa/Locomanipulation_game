"""The LSTM cell state c over a 180 s zero-command stand in MuJoCo (Isaac plant), per policy and actor: the gates
recomputed by hand from the exported weights (checked against the network's own h, c), per unit.
Writes cell_study.npz (per policy: c and f over time, mean over seeds) and prints a summary."""
import sys, glob, math, numpy as np, torch, yaml
from multiprocessing import Pool
sys.path.insert(0, "agent_scripts"); sys.path.insert(0, ".")
import hw_stress_mujoco as H

SECONDS = 180

class Probe:
    """The exported actor, stepped by hand: gates, c and h per step, and the network's action."""
    def __init__(self, net):
        self.net = net
        sd = {k: v.float() for k, v in net.state_dict().items()}
        self.W_ih, self.W_hh = sd["lstm.weight_ih_l0"], sd["lstm.weight_hh_l0"]
        self.b = sd["lstm.bias_ih_l0"] + sd["lstm.bias_hh_l0"]
        self.mean, self.std = sd["mean"], sd["std"]
        self.H = self.W_hh.shape[1]
        self.h = torch.zeros(1, 1, self.H); self.c = torch.zeros(1, 1, self.H)
        self.log = []
        self.err = 0.0
    def __call__(self, x):
        z = torch.clamp((x - self.mean) / (self.std + 1e-8), -5, 5)
        g = z @ self.W_ih.T + self.h[0] @ self.W_hh.T + self.b
        i, f, gg, o = g.chunk(4, dim=-1)
        i, f, gg, o = torch.sigmoid(i), torch.sigmoid(f), torch.tanh(gg), torch.sigmoid(o)
        c_hand = f * self.c[0] + i * gg
        a, h, c = self.net(x, self.h, self.c)
        self.err = max(self.err, (c[0] - c_hand).abs().max().item())
        self.h, self.c = h, c
        self.log.append(np.concatenate([c[0, 0].numpy(), f[0].numpy(), i[0].numpy(), o[0].numpy(), h[0, 0].numpy()]))
        return a
    def reset_state(self): pass
    def state_norm(self): return float(self.c.abs().mean())
    recurrent = True

def run(args):
    export, seed = args
    torch.set_num_threads(1)
    d = yaml.safe_load(open([p for p in glob.glob(export + "/*.yaml") if not p.endswith("export.yaml")][0]))
    d.update(yaml.safe_load(open(export + "/export.yaml")))
    actors = {a: Probe(torch.jit.load(f"{export}/{a}.pt").eval()) for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = H.StressMujoco(actors, dict(H.NOMINAL), d, "isaac", rng); sim.reset(rng); obs = sim.observations()
    fell = None
    with torch.no_grad():
        for k in range(int(SECONDS / H.DT)):
            obs = sim.step(obs)
            if sim.tilt() > 0.8 and fell is None:
                fell = k * H.DT; break
    out = {}
    for a, p in actors.items():
        L = np.array(p.log); n = len(L)
        full = np.full((int(SECONDS / H.DT), L.shape[1]), np.nan); full[:n] = L
        out[a] = full
    return out, fell, max(p.err for p in actors.values())

if __name__ == "__main__":
    pols = dict(a.split("=") for a in sys.argv[1:])
    res = {}
    with Pool(16) as pool:
        for name, export in pols.items():
            rs = pool.map(run, [(export, s) for s in range(4)])
            res[name] = rs
    T = np.arange(int(SECONDS / H.DT)) * H.DT
    npz = {}
    for name, rs in res.items():
        falls = [r[1] for r in rs if r[1] is not None]
        print(f"\n=== {name}: falls {len(falls)}/4 {falls}; hand-computed c vs network max diff {max(r[2] for r in rs):.1e}")
        for a in ("legs", "arms"):
            X = np.stack([r[0][a] for r in rs])            # seeds x steps x 5H
            Hn = X.shape[-1] // 5
            c, f, i, o, h = (X[..., k * Hn:(k + 1) * Hn] for k in range(5))
            cm = np.nanmean(c, axis=0)                     # steps x units, mean over seeds
            npz[f"{name}_{a}_c"] = cm; npz[f"{name}_{a}_f"] = np.nanmean(f, axis=0)
            at = lambda s: int(s / H.DT) - 1
            absc = np.nanmean(np.abs(c), axis=(0, 2))
            last = np.nanmean(np.abs(c[:, at(170):at(180)]), axis=(0, 1))            # per unit, last 10 s
            big = last > 2.0                                                          # tanh(c) > 0.96: h pinned at +-o
            fmean = np.nanmean(f[:, at(20):], axis=(0, 1))
            # clock units: c tracks time (|corr| > 0.9 over the stand, mean over seeds)
            ok = np.isfinite(cm[:, 0])
            corr = np.array([abs(np.corrcoef(T[ok], cm[ok, u])[0, 1]) if np.std(cm[ok, u]) > 1e-6 else 0 for u in range(Hn)])
            clock = (corr > 0.9) & (np.abs(cm[ok][-1] - cm[ok][at(20)]) > 0.5)
            sat = np.nanmean(np.abs(np.tanh(c)) > 0.99)
            print(f"  {a:4s} mean|c| 5s {absc[at(5)]:.2f}  20s {absc[at(20)]:.2f}  60s {absc[at(60)]:.2f}  180s {absc[at(180)]:.2f}"
                  f" | units |c|>2 at 180 s: {big.sum():2d}/{Hn} (their f {fmean[big].mean() if big.any() else float('nan'):.4f},"
                  f" others {fmean[~big].mean():.4f}) | clock units {clock.sum():2d} | tanh(c) saturated {100*sat:.0f}%"
                  f" | largest units {np.argsort(-last)[:3].tolist()} at |c| {np.round(np.sort(last)[::-1][:3], 1).tolist()}")
    np.savez_compressed(sys.argv[0].replace(".py", ".npz"), t=T, **npz)
