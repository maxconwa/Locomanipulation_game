"""Action smoothness in MuJoCo (Isaac plant): 20 s zero-command stand then 20 s walk at vx 0.3, 4 seeds per policy.
Per phase: leg joint-target change per policy step (rms, rad), leg raw-action change (rms), and the share of the
leg targets' power above 5 Hz (jitter)."""
import sys, glob, numpy as np, torch, yaml
from multiprocessing import Pool
sys.path.insert(0, "agent_scripts"); sys.path.insert(0, ".")
import hw_stress_mujoco as H

def run(args):
    export, seed = args
    torch.set_num_threads(1)
    d = yaml.safe_load(open([p for p in glob.glob(export + "/*.yaml") if not p.endswith("export.yaml")][0]))
    d.update(yaml.safe_load(open(export + "/export.yaml")))
    actors = {a: H.Recurrent(torch.jit.load(f"{export}/{a}.pt").eval()) for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = H.StressMujoco(actors, dict(H.NOMINAL), d, "isaac", rng); sim.reset(rng); obs = sim.observations()
    tg, ac = [], []
    with torch.no_grad():
        for k in range(int(40 / H.DT)):
            sim.vel_cmd = np.zeros(3) if k * H.DT < 20 else np.array([0.3, 0.0, 0.0])
            obs = sim.step(obs)
            tg.append(sim.applied[:12].copy()); ac.append(sim.raw_actions[:12].copy())
            if sim.tilt() > 0.8: return None
    return np.array(tg), np.array(ac)

def stats(tg, ac, a, b):
    t, x = tg[a:b], ac[a:b]
    dt_rms = np.sqrt((np.diff(t, axis=0) ** 2).mean())
    da_rms = np.sqrt((np.diff(x, axis=0) ** 2).mean())
    seg = t - t.mean(0)
    p = (np.abs(np.fft.rfft(seg, axis=0)) ** 2).sum(1); f = np.fft.rfftfreq(len(seg), H.DT)
    return dt_rms, da_rms, p[f > 5].sum() / p[f > 0.2].sum()

if __name__ == "__main__":
    pols = dict(a.split("=") for a in sys.argv[1:])
    with Pool(16) as pool:
        for name, export in pols.items():
            rs = [r for r in pool.map(run, [(export, s) for s in range(4)]) if r is not None]
            st = np.array([stats(tg, ac, int(5 / H.DT), int(20 / H.DT)) for tg, ac in rs])
            wk = np.array([stats(tg, ac, int(25 / H.DT), int(40 / H.DT)) for tg, ac in rs])
            print(f"{name:12s} runs {len(rs)}/4 | stand: target step {st[:,0].mean():.4f} rad, action step {st[:,1].mean():.3f}, >5 Hz {100*st[:,2].mean():.1f}%"
                  f" | walk: target step {wk[:,0].mean():.4f} rad, action step {wk[:,1].mean():.3f}, >5 Hz {100*wk[:,2].mean():.1f}%", flush=True)
