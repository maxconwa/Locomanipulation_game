"""Held-out check: the policy's block estimate error and its action's cosine with the teacher's on fresh episodes,
in envs the teacher drives (first half) and envs the policy drives (second half)."""
import argparse, os, sys
from isaaclab.app import AppLauncher
parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint"); parser.add_argument("--steps", type=int, default=400)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
app = AppLauncher(args).app
import gymnasium as gym, torch
import locomanipulation_game.tasks
from locomanipulation_game.tasks.locomanip_marl.vla_upper_env_cfg import VlaUpperEnvCfg
from locomanipulation_game.vla.teacher import StackTeacher
from locomanipulation_game.vla.encoders import FrozenEncoders
from locomanipulation_game.vla.model import VlaPolicy
from locomanipulation_game.vla.runner import PolicyRunner

def main():
    cfg = VlaUpperEnvCfg(); cfg.scene.num_envs = 64; cfg.seed = 7; cfg.sampler_floor = 1.0; cfg.training = False
    cfg.log_dir = os.path.dirname(args.checkpoint)
    env = gym.make("LocoManip-VlaUpper-v0", cfg=cfg).unwrapped
    n, dev = env.num_envs, env.device
    st = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    policy = VlaPolicy(**st["config"]).to(dev).eval(); policy.load_state_dict(st["policy"])
    enc = FrozenEncoders(dev)
    runner = PolicyRunner(env, enc, policy)
    teacher = StackTeacher(env)
    obs, _ = env.reset(); ev = torch.arange(n, device=dev)
    teacher.reset(ev); runner.reset(ev)
    student = ev >= n // 2
    legs = torch.zeros(n, env.cfg.action_spaces["legs"], device=dev)
    acc = {False: [], True: []}
    for step in range(args.steps):
        im = env.camera_images()
        runner.refresh_task(im["head_cam"])
        label = teacher.act()
        wrist = torch.stack([im["left_wrist_cam"], im["right_wrist_cam"]], 1)
        head = enc.head_small(im["head_cam"])
        if step % 4 == 0:
            with torch.no_grad():
                w = wrist.permute(0, 1, 4, 2, 3).float() / 255; h = head.permute(0, 3, 1, 2).float() / 255
                tok = enc.embed_wrists(w)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    mem, mask, est, direct = policy.memory(runner.vlm_tokens, runner.vlm_mask, tok, w, h, obs["arms"])
                    mean = policy.head.mean(mem, mask, direct).float()
                bp, _ = env.blocks_in_pelvis()
                true = torch.cat([bp[ev, env.pick_slot], bp[ev, env.target_slot]], 1)
                err = (est.float() - true).view(n, 2, 3)[..., :2].norm(dim=-1) * 100  # cm: pick, target
                a = mean[:, 0, :12].view(n, 2, 6)[ev, env.arm, :3]; t = label[:, :12].view(n, 2, 6)[ev, env.arm, :3]
                d = direct.float().view(n, -1, 14)[:, 0, :12].view(n, 2, 6)[ev, env.arm, :3]
                mv = t.norm(dim=1) > 0.05
                cos = torch.nn.functional.cosine_similarity(a, t, dim=1); cosd = torch.nn.functional.cosine_similarity(d, t, dim=1)
                for s in (False, True):
                    m = (student == s) & mv
                    acc[s].append(torch.stack([err[student == s, 0].mean(), err[student == s, 1].mean(), cos[m].mean(), cosd[m].mean(),
                                               a[m].norm(dim=1).mean(), t[m].norm(dim=1).mean()]))
        action = label.clone()
        action[student] = runner.act(wrist, head, obs["arms"], student)[student]
        obs, *_ = env.step({"legs": legs, "arms": action.clamp(-1, 1)}); runner.step()
        ended = env.ended.nonzero().flatten()
        if len(ended):
            teacher.reset(ended); runner.reset(ended)
    for s in (False, True):
        r = torch.stack(acc[s]).nanmean(0).tolist()
        print("[held]", "student-driven" if s else "teacher-driven", "pick_cm %.2f target_cm %.2f cos %.3f cos_state_branch %.3f |a| %.3f |t| %.3f" % tuple(r), flush=True)
try:
    main()
except Exception:
    import traceback; traceback.print_exc()
os._exit(0)
