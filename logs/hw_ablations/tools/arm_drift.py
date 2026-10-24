"""Over a 60 s zero-command stand in MuJoCo (Isaac plant): per 5 s window, the arms' residual action rms, the wrists'
distance from their rest targets, the legs' action rms and pelvis tilt, mean over seeds."""
import sys, numpy as np, torch
sys.path.insert(0, "agent_scripts"); sys.path.insert(0, ".")
import hw_stress_mujoco as H
import yaml, glob, os
from multiprocessing import Pool

def run(args):
    export, seed, reset_arms = args
    torch.set_num_threads(1)
    d = yaml.safe_load(open([p for p in glob.glob(export + "/*.yaml") if not p.endswith("export.yaml")][0]))
    d.update(yaml.safe_load(open(export + "/export.yaml")))
    f = dict(H.NOMINAL)
    actors = {a: H.Recurrent(torch.jit.load(f"{export}/{a}.pt").eval()) for a in ("legs", "arms")}
    rng = np.random.default_rng(seed)
    sim = H.StressMujoco(actors, f, d, "isaac", rng); sim.reset(rng); obs = sim.observations()
    rows = []
    with torch.no_grad():
        for k in range(int(60 / H.DT)):
            if reset_arms and k and k % 500 == 0: actors["arms"].reset_state()
            obs = sim.step(obs)
            wp = sim.wrist_poses_b()
            werr = np.linalg.norm(wp[:, :3] - sim.rest_b[:, :3], axis=1).mean()
            rows.append([np.sqrt(np.mean(sim.raw_actions[12:] ** 2)), werr, np.sqrt(np.mean(sim.raw_actions[:12] ** 2)),
                         sim.tilt(), actors["arms"].state_norm(), actors["legs"].state_norm()])
            if sim.tilt() > 0.8: break
    r = np.full((int(60 / H.DT), 6), np.nan); r[:len(rows)] = rows
    return r

if __name__ == "__main__":
    export = sys.argv[1]; reset = sys.argv[2] == "1"
    with Pool(8) as p: R = np.stack(p.map(run, [(export, s, reset) for s in range(8)]))
    W = int(5 / H.DT)
    print("window   arms|a|  wrist_err  legs|a|  tilt   arms|c|  legs|c|  alive")
    for w in range(12):
        seg = R[:, w * W:(w + 1) * W]
        print(f"{w*5:3d}-{w*5+5:3d}s " + "  ".join(f"{np.nanmean(seg[..., i]):7.3f}" for i in range(6)) + f"  {np.isfinite(seg[:, -1, 0]).sum()}/8")
