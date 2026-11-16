"""Train the upper-body vision-language-action policy of LocoManip-VlaUpper-v0 by DAgger from the scripted teacher.

    python scripts/vla/train.py --headless --enable_cameras --name Run1 [--num_envs 64] [--steps 40000]

Every env step the teacher (vla/teacher.py, which sees the blocks) labels every env's state with its action; the
policy's own inputs there (wrist images, the VLM's tokens of the head image and the instruction, proprioception) and
that label go into the replay (vla/buffer.py), and the action head takes `--grad_steps` gradient steps on batches of
it. Each episode is driven by the policy with probability p (0 for the first `--warmup` steps, then up to
`--student_max` over `--ramp` steps), else by the teacher with noise on its executed actions (the label stays the
clean one), so the replay holds the states the policy itself reaches and what to do there. Only the policy-driven
episodes' outcomes update the env's success-guided sampler (VlaUpperEnv.sampler).

The last `--scene_envs` envs are scene envs: the teacher drives them and their episode is ended every `--scene_steps`
steps, so each gives a newly drawn scene (blocks, colours, positions, instruction) that often. An episode lasts
some 350 steps, so without them the replay held a few hundred scenes, and the policy fitted those (0.7 cm block
position error on the replay, 2 to 10 cm on fresh episodes). Their first frame is also the held-out measure: the
policy's block estimate and its action against the teacher's on a scene before it is trained on (fresh_* in the log).

The policy runs as it would on the robot: the VLM is refreshed every `--vlm_every` steps, the action head queried
every `--replan` steps for a chunk of `--horizon` actions, of which the next `--replan` are executed.

The run goes to logs/vla/<time>_<name>/: log.jsonl (a line per `--log_every` steps), checkpoints/policy_<step>.pt (the
trained weights, the optimizer, the sampler), replay/ (the memmaps; deleted at the end unless --keep_replay).
"""

import argparse
import json
import os
import shutil
import time
from datetime import datetime

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--name", default="VlaUpper")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=40000, help="env steps (each of all the envs)")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--resume", default=None, help="a checkpoint to start from")
parser.add_argument("--warmup", type=int, default=600)
parser.add_argument("--ramp", type=int, default=4000)
parser.add_argument("--student_max", type=float, default=0.8)
parser.add_argument("--teacher_noise", type=float, default=0.04, help="std of the noise on the teacher's executed deltas")
parser.add_argument("--replay_steps", type=int, default=12000, help="env steps the replay holds")
parser.add_argument("--image_every", type=int, default=4, help="the replay stores (and samples) every this many steps' images")
parser.add_argument("--scene_envs", type=int, default=16, help="envs that only show new scenes (teacher-driven)")
parser.add_argument("--scene_steps", type=int, default=8, help="a scene env's episode length, steps")
parser.add_argument("--min_replay", type=int, default=300, help="env steps in the replay before training starts")
parser.add_argument("--grad_steps", type=float, default=1.0, help="gradient steps per env step")
parser.add_argument("--batch", type=int, default=192)
parser.add_argument("--lr", type=float, default=3e-4)
parser.add_argument("--horizon", type=int, default=16)
parser.add_argument("--head", default="flow", choices=("flow", "gaussian"), help="the action head (vla/model.py)")
parser.add_argument("--replan", type=int, default=4)
parser.add_argument("--vlm_every", type=int, default=8)
parser.add_argument("--log_every", type=int, default=100)
parser.add_argument("--save_every", type=int, default=2000)
parser.add_argument("--keep_replay", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.tasks.locomanip_marl.vla_upper_env_cfg import VlaUpperEnvCfg  # noqa: E402
from locomanipulation_game.vla.buffer import DaggerBuffer  # noqa: E402
from locomanipulation_game.vla.encoders import HEAD_SMALL, VLM_DIM, VLM_TOKENS, WORD_TOKENS, FrozenEncoders  # noqa: E402
from locomanipulation_game.vla.model import ACTION_DIM, PROPRIO_DIM, VlaPolicy, augment_images  # noqa: E402
from locomanipulation_game.vla.runner import PolicyRunner  # noqa: E402
from locomanipulation_game.vla.teacher import StackTeacher  # noqa: E402

TABLES = "logs/steerability/dev/policy/arm_target_tables_armsonly_v2.pt"  # the arm command's tables, built once


@torch.no_grad()
def fresh_scene_errors(policy, encoders, runner, wrist, head, proprio, blocks, label, scene, arm) -> torch.Tensor:
    """On the scene envs' new scenes, before any of them is in the replay: the policy's estimate's horizontal error
    of the instructed block and of its target (cm), and the cosine of its first position delta of the instructed
    arm with the teacher's."""
    ids = scene.nonzero(as_tuple=False).flatten()
    w = wrist[ids].permute(0, 1, 4, 2, 3).float() / 255.0
    h = head[ids].permute(0, 3, 1, 2).float() / 255.0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        memory, mask, estimate, direct = policy.memory(runner.vlm_tokens[ids], runner.vlm_mask[ids],
                                                       encoders.embed_wrists(w), w, h, proprio[ids])
        mean = policy.head.mean(memory, mask, direct).float()
    error = (estimate.float() - blocks[ids]).view(-1, 2, 3)[..., :2].norm(dim=-1).mean(dim=0) * 100.0
    rows = torch.arange(len(ids), device=ids.device)
    a = mean[:, 0, :12].view(-1, 2, 6)[rows, arm[ids], :3]
    t = label[ids, :12].view(-1, 2, 6)[rows, arm[ids], :3]
    return torch.cat([error, torch.nn.functional.cosine_similarity(a, t, dim=1).mean().view(1)])


def main():
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    run_dir = os.path.abspath(os.path.join("logs", "vla", datetime.now().strftime("%Y-%m-%d_%H-%M-%S_") + args_cli.name))
    os.makedirs(os.path.join(run_dir, "checkpoints"))
    if os.path.isfile(TABLES):
        shutil.copy(TABLES, run_dir)
    with open(os.path.join(run_dir, "args.json"), "w") as f:
        json.dump(vars(args_cli), f, indent=1)

    cfg = VlaUpperEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device
    cfg.seed = args_cli.seed
    cfg.log_dir = run_dir
    cfg.training = False  # the walking task's warm start and curriculum saves: not this task's
    env = gym.make("LocoManip-VlaUpper-v0", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device

    encoders = FrozenEncoders(dev)
    wrist_px = env.scene["left_wrist_cam"].cfg.height
    policy = VlaPolicy(wrist_tokens=(wrist_px // 14) ** 2, horizon=args_cli.horizon, head=args_cli.head).to(dev)
    optimizer = torch.optim.AdamW(policy.parameters(), lr=args_cli.lr, weight_decay=1e-4)
    start = 0
    if args_cli.resume:
        state = torch.load(args_cli.resume, map_location=dev, weights_only=False)
        # the weights whose shapes still match (a changed layer starts fresh, and then the optimizer does too)
        own = policy.state_dict()
        kept = {k: v for k, v in state["policy"].items() if k in own and own[k].shape == v.shape}
        for k, v in state["policy"].items():
            # a tensor that only grew (the VLM's position embeddings with a longer instruction): its old part
            if k in own and k not in kept and v.dim() == own[k].dim() and all(a <= b for a, b in zip(v.shape, own[k].shape)):
                own[k][tuple(slice(0, a) for a in v.shape)] = v
                print(f"[train] {k}: {tuple(v.shape)} of the checkpoint copied into {tuple(own[k].shape)}", flush=True)
        policy.load_state_dict(kept, strict=False)
        if len(kept) == len(own):
            optimizer.load_state_dict(state["optimizer"])
        else:
            print(f"[train] {len(own) - len(kept)} tensors not in the checkpoint or of another shape: fresh", flush=True)
        env.sampler.load_state_dict(state["sampler"])
        start = state["step"]
        print(f"[train] resumed {args_cli.resume} at step {start}", flush=True)
    print(f"[train] {sum(p.numel() for p in policy.parameters()) / 1e6:.1f} M trained parameters; run {run_dir}", flush=True)
    policy.eval()  # rollouts: dropout off, the state branch's running statistics
    runner = PolicyRunner(env, encoders, policy, replan=args_cli.replan, vlm_every=args_cli.vlm_every)
    teacher = StackTeacher(env)
    replay = DaggerBuffer(os.path.join(run_dir, "replay"), args_cli.replay_steps, n, (2, wrist_px, wrist_px, 3),
                          (*HEAD_SMALL, 3), PROPRIO_DIM, ACTION_DIM, (VLM_TOKENS + WORD_TOKENS, VLM_DIM),
                          args_cli.vlm_every, args_cli.horizon, args_cli.image_every)
    assert args_cli.scene_steps % args_cli.image_every == 0  # a scene's first frame is a stored one

    obs, _ = env.reset()
    everyone = torch.arange(n, device=dev)
    teacher.reset(everyone)
    runner.reset(everyone)
    legs = torch.zeros(n, env.cfg.action_spaces["legs"], device=dev)
    student = torch.zeros(n, dtype=torch.bool, device=dev)  # the envs whose episode the policy drives
    scene = everyone >= n - args_cli.scene_envs               # the scene envs
    fresh_parts: list[torch.Tensor] = []
    env.sampler_mask[:] = student
    episode = np.arange(n, dtype=np.int64)
    next_episode = n
    window = {"student": [], "teacher": []}  # the ended episodes' successes since the last log line
    losses: list[dict[str, float]] = []
    grad_debt, grad_total = 0.0, 0
    clock = time.time()
    log = open(os.path.join(run_dir, "log.jsonl"), "a")

    for step in range(start, args_cli.steps):
        p_student = args_cli.student_max * min(1.0, max(0.0, (step - args_cli.warmup) / max(args_cli.ramp, 1)))
        proprio = obs["arms"]
        images = env.camera_images()
        wrist = torch.stack([images["left_wrist_cam"], images["right_wrist_cam"]], dim=1)
        head = encoders.head_small(images["head_cam"])
        block_pos, _ = env.blocks_in_pelvis()
        blocks = torch.cat([block_pos[everyone, env.pick_slot], block_pos[everyone, env.target_slot]], dim=1)
        # the VLM's tokens where due (new ones go to the replay), then the policy's action where it drives
        fresh = runner.refresh_task(images["head_cam"])
        if fresh is not None:
            replay.add_vlm(fresh.cpu().numpy(), runner.vlm_tokens[fresh], runner.vlm_mask[fresh])
        label = teacher.act()
        action = label.clone()
        action[:, :12] += args_cli.teacher_noise * torch.randn_like(action[:, :12])
        if student.any():
            action[student] = runner.act(wrist, head, proprio, student)[student]
        if scene.any() and replay.count % args_cli.scene_steps == 0:
            fresh_parts.append(fresh_scene_errors(policy, encoders, runner, wrist, head, proprio, blocks, label, scene, env.arm))
        replay.add(wrist, head, proprio, label, blocks, episode)
        if replay.count % args_cli.scene_steps == 0:
            env.end_now[:] = scene
        obs, _, _, _, _ = env.step({"legs": legs, "arms": action.clamp(-1.0, 1.0)})
        runner.step()

        ended = env.ended.nonzero(as_tuple=False).flatten()
        if len(ended):
            for e in ended.tolist():
                if not scene[e]:
                    window["student" if student[e] else "teacher"].append(float(env.ended_success[e]))
            teacher.reset(ended)
            runner.reset(ended)
            student[ended] = torch.rand(len(ended), device=dev) < p_student
            student[scene] = False
            env.sampler_mask[:] = student
            ids = ended.cpu().numpy()
            episode[ids] = np.arange(next_episode, next_episode + len(ids))
            next_episode += len(ids)

        if replay.count >= args_cli.min_replay:
            grad_debt += args_cli.grad_steps
            policy.train()
            while grad_debt >= 1.0:
                grad_debt -= 1.0
                batch = replay.sample(args_cli.batch, dev)
                wrist_f = augment_images(batch["wrist"].permute(0, 1, 4, 2, 3).float() / 255.0)
                head_f = augment_images(batch["head"].permute(0, 3, 1, 2).float().unsqueeze(1) / 255.0).squeeze(1)
                tokens = encoders.embed_wrists(wrist_f)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss, parts = policy.loss(batch["vlm"], batch["vlm_mask"], tokens, wrist_f, head_f, batch["proprio"],
                                              batch["target"], batch["valid"], batch["blocks"])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                optimizer.step()
                grad_total += 1
                losses.append(parts)
            policy.eval()

        if (step + 1) % args_cli.log_every == 0:
            row = {"step": step + 1, "env_steps": (step + 1) * n, "grad_steps": grad_total, "p_student": round(p_student, 3),
                   "steps_per_s": round(args_cli.log_every / (time.time() - clock), 2),
                   "student_episodes": len(window["student"]), "teacher_episodes": len(window["teacher"]),
                   "student_success": float(np.mean(window["student"])) if window["student"] else None,
                   "teacher_success": float(np.mean(window["teacher"])) if window["teacher"] else None,
                   "sampler_rate_mean": float(env.sampler.rate.mean()),
                   "sampler_weight_max": float(env.sampler.weights().max() * env.sampler.rate.numel())}
            if fresh_parts:
                fresh = torch.stack(fresh_parts).nanmean(dim=0).tolist()
                row.update({"fresh_pick_cm": fresh[0], "fresh_target_cm": fresh[1], "fresh_cosine": fresh[2]})
                fresh_parts = []
            if losses:
                row.update({key: float(np.mean([part[key] for part in losses])) for key in losses[0]})
            log.write(json.dumps(row) + "\n")
            log.flush()
            print("[train] " + json.dumps(row), flush=True)
            window = {"student": [], "teacher": []}
            losses = []
            clock = time.time()
        if (step + 1) % args_cli.save_every == 0 or step + 1 == args_cli.steps:
            torch.save({"policy": policy.state_dict(), "optimizer": optimizer.state_dict(), "config": policy.config,
                        "sampler": env.sampler.state_dict(), "step": step + 1, "args": vars(args_cli)},
                       os.path.join(run_dir, "checkpoints", f"policy_{step + 1}.pt"))
    log.close()
    if not args_cli.keep_replay:
        shutil.rmtree(os.path.join(run_dir, "replay"), ignore_errors=True)


try:
    main()
except Exception:
    import traceback

    traceback.print_exc()
os._exit(0)  # simulation_app.close() hangs after camera renders
