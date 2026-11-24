"""Fit PULSE's latent prior on a stage-1 run: proprio history -> the stage-1 latent.

  python -m pretraining.scripts.fit_prior data/pretraining/runs/isaac_s1_golem_aware \\
      [--out <file>] [--num-envs 2048] [--iterations 6000] [--hidden-dims 512 256] [--hide-last-action]

Ported from h12_train (this repository's own code, branch supermimic, commit b175f1c), ``scripts/fit_prior.py``.
Changes: Isaac Lab's ``LocoManip-Pretrain-Track-v0`` env in place of mjlab's (on the run's own clips, from its
``run.json``, frames drawn uniformly, the env's own randomisation: no ``--randomisation`` options); argparse in place
of tyro; ``--out`` defaults to ``<stage-1 run>/prior.pt``.

Runs the frozen stage-1 actors on the clips in the sim and fits, per agent, an MLP (``actors.LatentPrior``) from the
normalised proprio history to the latent the stage-1 encoder gives the clip's goal there: one Adam step per sim step
on ``mean((tanh(prior(history)) - z)^2)`` over all envs. Writes ``{agent: state dict}`` to ``--out`` and a ``.json``
beside it with the settings and, per agent, the last 500 steps' mean squared error, the latent's variance and the
share of it the prior explains (``explained``). ``LocoManip-Marl-Pulse-v0`` reads the file (its ``pulse_prior``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path

TASK = "LocoManip-Pretrain-Track-v0"
WINDOW = 500  # steps the report averages over


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("stage1", type=Path, help="the stage-1 run directory (actors.pt, run.json)")
    p.add_argument("--out", type=Path, default=None, help="default: <stage1>/prior.pt")
    p.add_argument("--num-envs", type=int, default=2048)  # stage 1's, on its 13,578 clips
    p.add_argument("--iterations", type=int, default=6000)
    p.add_argument("--hidden-dims", type=int, nargs="+", default=[512, 256])
    p.add_argument("--lr", type=float, default=1.0e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hide-last-action", action="store_true", help="zero the last action in the prior's input")
    return p


def explained(mse: list[float], var: list[float]) -> dict[str, float]:
    """The report's numbers over the last ``WINDOW`` steps: mean squared error, latent variance, ``1 - mse / var``."""
    m = sum(mse[-WINDOW:]) / len(mse[-WINDOW:])
    v = sum(var[-WINDOW:]) / len(var[-WINDOW:])
    return {"mse": m, "latent_variance": v, "explained": 1.0 - m / v if v > 0.0 else float("nan")}


def fit(args: argparse.Namespace, device: str) -> dict:
    """Collect with the stage-1 actors and fit both priors; write the files and return the report. Needs the app."""
    import gymnasium as gym
    import torch
    from isaaclab_tasks.utils import load_cfg_from_registry

    import pretraining.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.actors import LatentPrior
    from pretraining.marl.config import AGENTS
    from pretraining.marl.runner import ImitationRunner, TwoAgentRunnerCfg
    from pretraining.marl.runs import atomic_save, config_from_dict, from_repo_path
    from pretraining.supermimic.runners.base_runner import EnvAdapter

    os.environ.setdefault("WANDB_MODE", "offline")
    torch.manual_seed(args.seed)
    stage1 = args.stage1.resolve()
    out = (args.out or stage1 / "prior.pt").resolve()
    meta = json.loads((stage1 / "run.json").read_text())
    cfg = config_from_dict(TwoAgentRunnerCfg, meta["runner"])
    cfg.sampling_mode, cfg.buffer_size = "uniform", 8
    files = [str(from_repo_path(f)) for f in meta["motion_files"]]
    env_cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = device
    env_cfg.seed = args.seed
    env_cfg.commands.motion.motion_file = files
    env_cfg.commands.motion.sampling_mode = "uniform"
    env = EnvAdapter(gym.make(TASK, cfg=env_cfg).unwrapped)
    runner = ImitationRunner(env, cfg, rewards=None, run_dir=None, motion_files=files, device=device)
    state = torch.load(stage1 / "actors.pt", map_location=device, weights_only=False)
    for a in AGENTS:
        runner.actors[a].load_state_dict(state[a])
        runner.actors[a].eval()
    priors = {
        a: LatentPrior(runner.obs_dim, cfg.history_length, tuple(args.hidden_dims), cfg.latent_dim, env.num_actions,
                       args.hide_last_action).to(device)
        for a in AGENTS
    }
    opt = torch.optim.Adam([p for m in priors.values() for p in m.parameters()], args.lr)
    rep, inner = runner._representation, env.unwrapped
    runner.env.reset()
    runner._seed_sim_history()
    mse = {a: [] for a in AGENTS}
    var = {a: [] for a in AGENTS}
    for it in range(args.iterations):
        with torch.inference_mode():
            obs = runner.env.get_observations()
            full = obs["full_state"].to(device)
            rep_state, aux = rep.represent(full), rep.extract_aux_state(full)
            sensor = rep.compute_actor_sensor(
                rep_state, runner._sim_context.read(inner), aux_state=aux, sim_actor_obs=obs["actor"].to(device),
                noisy=True,
            )
            history = runner._collection_history(sensor)
            goals = runner._goals(runner.motion_cmd.time_steps.to(device).clone(), aux)
            actions, _ = runner.act(history, goals)
            inputs, targets = {}, {}
            for a in AGENTS:
                actor = runner.actors[a]
                inputs[a] = actor.obs_normalizer(history).flatten(1)
                goal = actor.goal_normalizers["motion"](goals[a]["goal_motion"])
                targets[a] = torch.tanh(actor.encoders["motion"](goal))
            next_obs, _, dones, _ = runner.env.step(actions)
            runner._advance_sim_history(next_obs["actor"].to(device), dones)
        loss = 0.0
        for a in AGENTS:
            x, z = inputs[a].clone(), targets[a].clone()  # out of inference mode, for autograd
            err = ((torch.tanh(priors[a](x)) - z) ** 2).mean()
            loss = loss + err
            mse[a].append(float(err))
            var[a].append(float(z.var(0).mean()))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if it % 250 == 0 or it == args.iterations - 1:
            print(f"[fit_prior] {it}: " + ", ".join(
                f"{a} mse {sum(mse[a][-250:]) / len(mse[a][-250:]):.4f} "
                f"(latent variance {sum(var[a][-250:]) / len(var[a][-250:]):.4f})" for a in AGENTS), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_save({a: priors[a].state_dict() for a in AGENTS}, out)
    report = {
        "stage1": str(stage1),
        "iterations": args.iterations,
        "num_envs": args.num_envs,
        "hidden_dims": list(args.hidden_dims),
        "lr": args.lr,
        "seed": args.seed,
        "hide_last_action": args.hide_last_action,
        **{a: explained(mse[a], var[a]) for a in AGENTS},
    }
    out.with_suffix(".json").write_text(json.dumps(report, indent=1))
    print(f"[fit_prior] {out}: {json.dumps({a: report[a] for a in AGENTS})}", flush=True)
    return report


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    stage1 = args.stage1.resolve()
    for name in ("actors.pt", "run.json"):
        if not (stage1 / name).is_file():
            print(f"[fit_prior] error: {stage1} has no {name}", file=sys.stderr)
            sys.exit(2)
    args.headless = True
    app = AppLauncher(args).app
    try:
        fit(args, args.device or "cuda:0")
    except BaseException:
        # as train.py: a failed run can leave the app unable to shut down
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    # not app.close(): after a fit it did not return (the sim test's subprocess, results written, was killed at its
    # 900 s limit). The prior and its report are on disk, so leave without closing the app.
    os._exit(0)


if __name__ == "__main__":
    main()
