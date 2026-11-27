"""Can the frozen stage-1 decoders of LocoManip-Marl-Pulse-v0 hold the wrists at the task's goal poses at all?

  python -m pretraining.scripts.reach_check [--stage1-run <run>] [--nominal] [--goals 16] [--generations 25]
      [--episode-s 3] [--legs-too] [--seed 0] [--out <json>]

No learned policy: a cross-entropy search, in the simulator, over a constant latent residual u per candidate (the env's
action, z = tanh(u + prior(proprio)); the arms' 32, with --legs-too the legs' 32 as well, else the legs' u = 0), held
for a whole episode (the decoder is proprio-conditioned, so this is a closed-loop behaviour, but a restricted class).
num_envs = goals x candidates: each group of candidates shares one wrist goal per arm, drawn by the arm_targets command
term's own sampler as the arms scorecard draws standing goals (arm goals only, no off-table, no rest goals, velocity
zero), copied to the group after every reset and never resampled during an episode. Fitness per candidate: the mean
over the episode's last 0.5 s of max(pos_err / 5 cm, rot_err / 20 deg), the worse arm, with the env's own errors
(ArmTargetsCommand.errors(), what the reward terms and the scorecard read); a candidate whose episode ended (a fall)
gets FALL_PENALTY. Per goal the top ELITE share refits the mean and std (std floored). All envs are reset before every
generation (the env's reset randomisation). Afterwards each goal's best u, the final CEM mean and u = 0 are re-run on
fresh resets across the group.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback

TASK = "LocoManip-Marl-Pulse-v0"
POS_TOL, ROT_TOL = 0.05, math.radians(20.0)
FALL_PENALTY = 1.0e3
ELITE = 0.1
STD_FLOOR = 0.05
CLIP = 5.0
TAIL_S = 0.5
GOAL_KEYS = ("believed_b", "seed_q", "goal_height", "target_drop", "needs_crouch")


def run(args) -> dict:
    import gymnasium as gym
    import torch
    from isaaclab.utils.math import quat_error_magnitude
    from isaaclab_tasks.utils import load_cfg_from_registry

    import locomanipulation_game.tasks  # noqa: F401
    from locomanipulation_game.tasks.locomanip_marl.mdp.commands import _apply

    G, P = args.goals, args.candidates
    cfg = load_cfg_from_registry(TASK, "env_cfg_entry_point")
    cfg.scene.num_envs = G * P
    cfg.seed = args.seed
    cfg.training = False  # goals enabled from the start, no warm start, no state saving (as evaluation_cfgs)
    if args.stage1_run:
        cfg.pulse_stage1_run = args.stage1_run
    if args.nominal:  # as pulse_baseline: no pre-pose resets, gain draws, action delay, encoder offsets
        cfg.events.prepose = None
        cfg.events.actuator_scales = None
        cfg.action_delay_range = None
        cfg.encoder_offset_rad = None
    # as the arms scorecard: arm goals only, none resampled during an episode, standing
    cfg.episode_length_s = 1.0e4
    cfg.episode_length_range_s = None
    arm_cfg, vel_cfg = cfg.commands.arm_targets, cfg.commands.base_velocity
    arm_cfg.arm_goal_prob = 1.0
    arm_cfg.resampling_time_range, arm_cfg.nav_time_range = (1.0e6, 1.0e6), (1.0e6, 1.0e6)
    arm_cfg.debug_vis = vel_cfg.debug_vis = False
    vel_cfg.resampling_time_range, vel_cfg.rel_standing_envs = (1.0e6, 1.0e6), 1.0
    cfg.sync_derived()
    env = gym.make(TASK, cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    arm = env.command_manager.get_term("arm_targets")
    vel = env.command_manager.get_term("base_velocity")
    robot = env.scene["robot"]
    arm._judge_kernels = lambda ended: None
    arm.goals_enabled, arm.goals_enabled_at = True, None
    arm.cfg.offtable_prob = 0.0
    arm.cfg.rest_goal_prob = 0.0
    assert arm.cfg.pelvis_fixed_goals, "expected Marl-Direct's pelvis-fixed goals (the goal is believed_b)"
    group = torch.arange(n, device=dev) // P
    steps = int(round(args.episode_s / env.step_dt))
    tail = int(round(TAIL_S / env.step_dt))
    dims = {"legs": env.cfg.action_spaces["legs"], "arms": env.cfg.action_spaces["arms"]}
    search_dim = dims["arms"] + (dims["legs"] if args.legs_too else 0)
    torch.manual_seed(args.seed)

    with torch.inference_mode():
        env.reset()
        # the G goals: the command term's own sampler, on one env per group (as the scorecard's arm._resample)
        firsts = torch.arange(G, device=dev) * P
        arm._resample(firsts)
        goals = {k: getattr(arm, k)[firsts].clone() for k in GOAL_KEYS if torch.is_tensor(getattr(arm, k, None))}
        assert bool(arm.arm_mode[firsts].all())

        def set_goals():
            for k, v in goals.items():
                getattr(arm, k)[:] = v[group]
            arm.arm_mode[:] = True
            arm.time_left[:] = 1.0e6
            arm.goal_reached[:] = False
            arm.hold_time[:] = 0.0
            arm.invalidate_errors()
            vel.vel_command_b[:] = 0.0
            vel.vel_command_filtered[:] = 0.0

        check = {"goal_asserts": 0, "error_max_abs_diff_m": 0.0, "error_max_abs_diff_rad": 0.0,
                 "vel_cmd_max_abs": 0.0}

        def episode(u: torch.Tensor, record_check: bool = False):
            """Every env runs its u (n, search_dim) for one episode from a fresh reset. Returns (fitness (n,), fell (n,),
            tail pos err (n, 2) m, tail rot err (n, 2) rad)."""
            env.reset()
            set_goals()
            actions = {"legs": torch.zeros(n, dims["legs"], device=dev), "arms": u[:, :dims["arms"]].clamp(-CLIP, CLIP)}
            if args.legs_too:
                actions["legs"] = u[:, dims["arms"]:].clamp(-CLIP, CLIP)
            fell = torch.zeros(n, dtype=torch.bool, device=dev)
            pos_t = torch.zeros(n, 2, device=dev)
            rot_t = torch.zeros(n, 2, device=dev)
            score = torch.zeros(n, device=dev)
            for k in range(steps):
                _, _, terminated, truncated, _ = env.step(actions)
                fell |= terminated["legs"] | truncated["legs"] | terminated["arms"] | truncated["arms"]
                ok = ~fell
                # (a) the goal each still-running env tracks is its group's, and the velocity command zero
                same = torch.isclose(arm.believed_b[ok], goals["believed_b"][group[ok]], atol=1e-6).all()
                if not (bool(same) and bool(arm.arm_mode[ok].all())):
                    raise AssertionError(f"step {k}: an env's goal is not its group's")
                check["goal_asserts"] += 1
                check["vel_cmd_max_abs"] = max(check["vel_cmd_max_abs"], float(vel.vel_command_b[ok].abs().max()))
                pe, re_ = arm.errors()
                if record_check and k in (0, steps // 2, steps - 1):
                    # (b) the env's errors against our own from the robot's wrist bodies and the group's pelvis-frame goal
                    tw = _apply(robot.data.root_pos_w, robot.data.root_quat_w, goals["believed_b"][group])
                    bp = robot.data.body_pos_w[:, arm.body_ids]
                    bq = robot.data.body_quat_w[:, arm.body_ids]
                    my_pe = torch.norm(bp - tw[..., :3], dim=-1)
                    my_re = quat_error_magnitude(bq.reshape(-1, 4), tw[..., 3:].reshape(-1, 4)).view(n, 2)
                    check["error_max_abs_diff_m"] = max(check["error_max_abs_diff_m"], float((my_pe - pe)[ok].abs().max()))
                    check["error_max_abs_diff_rad"] = max(check["error_max_abs_diff_rad"],
                                                          float((my_re - re_)[ok].abs().max()))
                if k >= steps - tail:
                    pos_t += pe
                    rot_t += re_
                    score += torch.maximum(pe / POS_TOL, re_ / ROT_TOL).max(dim=1).values
            pos_t /= tail
            rot_t /= tail
            score = torch.where(fell, FALL_PENALTY, score / tail)
            return score, fell, pos_t, rot_t

        # -- the cross-entropy search, per goal
        mean = torch.zeros(G, search_dim, device=dev)
        std = torch.ones(G, search_dim, device=dev)
        n_elite = max(2, int(round(ELITE * P)))
        best_f = torch.full((G,), float("inf"), device=dev)
        best_u = torch.zeros(G, search_dim, device=dev)
        best_pos = torch.zeros(G, 2, device=dev)
        best_rot = torch.zeros(G, 2, device=dev)
        traj = {"best": [], "elite_mean": [], "pop_median": [], "fell_share": [], "std_mean": []}
        zero_ref = None
        t0 = time.time()
        for gen in range(args.generations):
            u = mean[group] + std[group] * torch.randn(n, search_dim, device=dev)
            u = u.clamp(-CLIP, CLIP)
            u[firsts] = 0.0 if gen == 0 else mean  # slot 0: u = 0 at first (the reference), then the current mean
            f, fell, pos, rot = episode(u, record_check=(gen == 0))
            if gen == 0:
                zero_ref = {"pos_mm": (pos[firsts] * 1000).tolist(), "rot_deg": torch.rad2deg(rot[firsts]).tolist(),
                            "fell": fell[firsts].tolist(), "fitness": f[firsts].tolist()}
            fg = f.view(G, P)
            order = fg.argsort(dim=1)
            elite_idx = order[:, :n_elite] + torch.arange(G, device=dev).unsqueeze(1) * P
            elite_u = u[elite_idx]  # (G, n_elite, D)
            mean = elite_u.mean(1)
            std = elite_u.std(1).clamp(min=STD_FLOOR)
            top = order[:, 0] + torch.arange(G, device=dev) * P
            better = f[top] < best_f
            best_f = torch.where(better, f[top], best_f)
            best_u[better] = u[top][better]
            best_pos[better] = pos[top][better]
            best_rot[better] = rot[top][better]
            traj["best"].append(fg.min(1).values.tolist())
            traj["elite_mean"].append(f[elite_idx].mean(1).tolist())
            traj["pop_median"].append(fg.median(1).values.tolist())
            traj["fell_share"].append(fell.view(G, P).float().mean(1).tolist())
            traj["std_mean"].append(std.mean(1).tolist())
            print(f"[reach_check] gen {gen}: best fitness median over goals {float(fg.min(1).values.median()):.3f}, "
                  f"elite mean {float(f[elite_idx].mean()):.3f}, fell {float(fell.float().mean()):.1%}, "
                  f"std {float(std.mean()):.3f}, best-ever median {float(best_f.median()):.3f} ({time.time() - t0:.0f} s)",
                  flush=True)

        # -- re-evaluation on fresh resets: half of each group the best u, a quarter the final mean, a quarter u = 0
        slot = torch.arange(n, device=dev) % P
        which = torch.where(slot < P // 2, 0, torch.where(slot < 3 * P // 4, 1, 2))
        u = torch.where((which == 0).unsqueeze(1), best_u[group], torch.where((which == 1).unsqueeze(1), mean[group], 0.0))
        f, fell, pos, rot = episode(u)
        reeval = {}
        for wi, name in enumerate(("best_u", "final_mean", "zero_u")):
            per = []
            for g in range(G):
                m = (group == g) & (which == wi)
                ok = m & ~fell
                pe, re_ = pos[ok], rot[ok]  # (k, 2)
                worst_ok = ((pe < POS_TOL) & (re_ < ROT_TOL)).all(1)
                per.append({
                    "n": int(m.sum()), "fell_share": float(fell[m].float().mean()),
                    "pos_mm_mean": (pe.mean(0) * 1000).tolist() if len(pe) else None,
                    "pos_mm_median": (pe.median(0).values * 1000).tolist() if len(pe) else None,
                    "rot_deg_mean": torch.rad2deg(re_.mean(0)).tolist() if len(pe) else None,
                    "rot_deg_median": torch.rad2deg(re_.median(0).values).tolist() if len(pe) else None,
                    "within_5cm_20deg_share": float(worst_ok.float().mean()) if len(pe) else None,
                    "fitness_median": float(f[ok].median()) if len(pe) else None})
            reeval[name] = per

    def share(mask):
        return float(sum(mask) / len(mask))

    bp, br = (best_pos * 1000).cpu(), torch.rad2deg(best_rot).cpu()
    both = [bool((bp[g] < 50).all() and (br[g] < 20).all()) for g in range(G)]
    pos5 = [bool((bp[g] < 50).all()) for g in range(G)]
    pos10 = [bool((bp[g] < 100).all()) for g in range(G)]
    import statistics as st

    def re_summary(name):
        rows = [r for r in reeval[name] if r["pos_mm_median"] is not None]
        return {"median_over_goals_of_worse_arm_pos_mm_median": st.median(max(r["pos_mm_median"]) for r in rows),
                "median_over_goals_of_worse_arm_rot_deg_median": st.median(max(r["rot_deg_median"]) for r in rows),
                "goals_with_median_rollout_within_5cm_20deg": share(
                    [max(r["pos_mm_median"]) < 50 and max(r["rot_deg_median"]) < 20 for r in rows]),
                "mean_within_5cm_20deg_share": st.mean(r["within_5cm_20deg_share"] for r in rows),
                "fell_share": st.mean(r["fell_share"] for r in reeval[name])}

    zr_pos = torch.tensor(zero_ref["pos_mm"])
    zr_rot = torch.tensor(zero_ref["rot_deg"])
    out = {
        "task": TASK, "stage1_run": env._pulse_shared.stage1_run, "nominal": args.nominal, "legs_too": args.legs_too,
        "goals": G, "candidates": P, "generations": args.generations, "episode_s": args.episode_s, "seed": args.seed,
        "tail_s": TAIL_S, "elite": ELITE, "std_floor": STD_FLOOR, "fall_penalty": FALL_PENALTY,
        "summary": {
            "best_within_5cm_20deg_both_arms": share(both), "best_within_5cm": share(pos5),
            "best_within_10cm": share(pos10),
            "best_pos_mm_median_worse_arm": float(bp.max(1).values.median()),
            "best_rot_deg_median_worse_arm": float(br.max(1).values.median()),
            "zero_u_pos_mm_median_worse_arm": float(zr_pos.max(1).values.median()),
            "zero_u_rot_deg_median_worse_arm": float(zr_rot.max(1).values.median()),
            "zero_u_within_5cm_20deg": share([bool((zr_pos[g] < 50).all() and (zr_rot[g] < 20).all()) for g in range(G)]),
            "reeval": {k: re_summary(k) for k in reeval},
        },
        "per_goal": [{
            "goal_b": goals["believed_b"][g].tolist(), "goal_height": float(goals["goal_height"][g]),
            "best_fitness": float(best_f[g]), "best_pos_mm": bp[g].tolist(), "best_rot_deg": br[g].tolist(),
            "best_u": best_u[g].tolist(),
            "zero_u": {"pos_mm": zero_ref["pos_mm"][g], "rot_deg": zero_ref["rot_deg"][g], "fell": zero_ref["fell"][g]},
            "fitness_best_per_gen": [t[g] for t in traj["best"]],
            "fitness_elite_mean_per_gen": [t[g] for t in traj["elite_mean"]],
            "fitness_pop_median_per_gen": [t[g] for t in traj["pop_median"]],
            "fell_share_per_gen": [t[g] for t in traj["fell_share"]],
            "std_mean_per_gen": [t[g] for t in traj["std_mean"]],
            "reeval": {k: v[g] for k, v in reeval.items()},
        } for g in range(G)],
        "sanity": check,
    }
    print("[reach_check] summary " + json.dumps(out["summary"], indent=1), flush=True)
    print("[reach_check] sanity " + json.dumps(check), flush=True)
    return out


def main() -> None:
    from isaaclab.app import AppLauncher

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--stage1-run", default="data/pretraining/runs/main_s1_finetune_long")
    p.add_argument("--nominal", action="store_true", help="without the pre-pose resets, gain draws, delay, offsets")
    p.add_argument("--goals", type=int, default=16)
    p.add_argument("--candidates", type=int, default=256)
    p.add_argument("--generations", type=int, default=25)
    p.add_argument("--episode-s", type=float, default=3.0)
    p.add_argument("--legs-too", action="store_true", help="search the legs' residual too (64-d)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default=None, help="write the result as json here")
    AppLauncher.add_app_launcher_args(p)
    args = p.parse_args()
    args.headless = True
    AppLauncher(args)
    try:
        out = run(args)
        if args.out:
            os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
            with open(args.out, "w") as f:
                json.dump(out, f, indent=1)
    except BaseException:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
