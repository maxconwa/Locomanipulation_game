"""Check the env's world-model hooks (LocoManipMarlEnv.world_state, sim_context, goal_state, task_rewards,
obs_noise_scale; goals.py, sim_context.py) on random actions.

    python scripts/supermimic/check_env_hooks.py --headless [--task LocoManip-LegsOnly-v0] [--num_envs 64] [--steps 300]

Prints, and fails (exit 1) on a mismatch:
- the `world` group's size and term layout, its min / max per term over the run; that its terms shared with the
  critic's group (pelvis height, gravity, velocities, joints) equal them exactly; the legs / arms / critic sizes;
- sim_context's size, that it changes only for the envs that just reset, and that the recovered gain x strength is
  each actuator's stiffness over its nominal kp;
- advance_goal against the env: for the envs that neither reset nor had a command resample between two steps,
  advance_goal(goal_state_t) against goal_state_t+1 (max abs diff per part); goal_features against the legs'
  observed velocity_commands, gait_phase and gait_phase_cos (exactly);
- task_rewards[agent] + termination_penalty * terminated == the returned reward (exactly); task_rewards less
  golem_estop_penalty * trip_events is the reward manager's, floored at 0 (to float rounding);
- obs_noise_scale per term of each actor's observation.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="LocoManip-LegsOnly-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--action_std", type=float, default=0.5)
parser.add_argument("--seed", type=int, default=0)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl import goals  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.sim_context import SIM_CONTEXT_TERMS  # noqa: E402

EXPECTED_SIZES = {"LocoManip-LegsOnly-v0": {"legs": 79, "arms": 122, "critic": 137}}
SHARED = ("pelvis_height", "projected_gravity", "base_lin_vel", "base_ang_vel", "joint_pos", "joint_vel")
failures = []


def check(ok: bool, what: str):
    print(f"  [{'ok' if ok else 'FAIL'}] {what}")
    if not ok:
        failures.append(what)


def layout(manager, group: str) -> dict[str, slice]:
    out, start = {}, 0
    for name, dims in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
        size = int(torch.tensor(dims).prod())
        out[name] = slice(start, start + size)
        start += size
    return out


torch.manual_seed(args.seed)
cfg = parse_env_cfg(args.task, device=args.device or "cuda:0", num_envs=args.num_envs)
cfg.seed = args.seed
env = gym.make(args.task, cfg=cfg).unwrapped
env.reset()
manager = env.observation_manager
n, dev = env.num_envs, env.device
print(f"\n===== {args.task}: {n} envs, {args.steps} steps of random actions (std {args.action_std}) =====")

# -- sizes and layouts
dims = {g: manager.group_obs_dim[g][0] for g in ("legs", "arms", "critic", "world")}
print(f"group sizes: {dims}; spaces: obs {env.cfg.observation_spaces}, state {env.cfg.state_space}")
world, critic, legs_obs = layout(manager, "world"), layout(manager, "critic"), layout(manager, "legs")
print("world layout (S = %d):" % dims["world"])
for name, s in world.items():
    print(f"  {name:20s} [{s.start:3d}:{s.stop:3d}] {s.stop - s.start}")
expected = EXPECTED_SIZES.get(args.task)
if expected:
    check(all(dims[g] == expected[g] for g in expected), f"legs/arms/critic sizes {[dims[g] for g in expected]} == "
          f"{list(expected.values())}")
check(env.cfg.observation_spaces == {a: dims[a] for a in env.cfg.possible_agents} and env.cfg.state_space == dims["critic"],
      "the env's spaces are the agents' and the critic's groups (world not among them)")
check(env.world_state().shape == (n, dims["world"]), f"world_state() shape {tuple(env.world_state().shape)}")

# -- sim context
xi = env.sim_context().clone()
print(f"sim_context C = {xi.shape[1]} ({SIM_CONTEXT_TERMS})")
check(xi.shape[1] == sum(SIM_CONTEXT_TERMS.values()), "C matches SIM_CONTEXT_TERMS")
ctx = env._sim_context
ids = torch.arange(n, device=dev)
gain, strength = ctx.actuator_scales(ids)
err = 0.0
for name, (kp, _, _) in getattr(env, "_actuator_nominal", {}).items():
    if name in ctx._columns:
        cols, places = ctx._columns[name]
        ratio = env.scene["robot"].actuators[name].stiffness[:, cols] / kp[:, cols]
        err = max(err, ((gain[:, places] * strength[:, places] - ratio).abs() / ratio.abs().clamp(min=1e-9)).max().item())
check(err < 1e-5, f"gain x strength == stiffness / nominal kp: max rel diff {err:.2e}")
print(f"  gain range [{gain.min():.3f}, {gain.max():.3f}], strength [{strength.min():.3f}, {strength.max():.3f}]")
offsets = {}
start = 0
for name, size in SIM_CONTEXT_TERMS.items():
    offsets[name] = slice(start, start + size)
    start += size
for name, s in offsets.items():
    print(f"  xi {name:20s} [{xi[:, s].min():+.3f}, {xi[:, s].max():+.3f}]")

# -- goals
specs = {a: goals.goal_spec(env, a) for a in env.cfg.possible_agents}
for agent, spec in specs.items():
    print(f"goal {agent}: Gs {spec.state_dim} {spec.state_layout}, Gf {spec.feature_dim} {spec.feature_layout}")
vel = env.command_manager.get_term(goals.VELOCITY_COMMAND)
arm = env.command_manager.get_term(goals.ARM_COMMAND)
gstate = {a: env.goal_state(a) for a in specs}
check(all(gstate[a].shape == (n, specs[a].state_dim) for a in specs), "goal_state shapes")

# -- noise
for agent in env.cfg.possible_agents:
    widths = env.obs_noise_scale(agent)
    terms = ", ".join(f"{name} {s.stop - s.start}x{widths[s].max().item():g}" for name, s in layout(manager, agent).items())
    print(f"obs_noise_scale {agent} ({widths.numel()}): {terms}")
    check(widths.numel() == dims[agent], f"obs_noise_scale({agent}) size")

# -- the run
shared_err = {k: 0.0 for k in SHARED}
xi_bad, xi_reset_changed, xi_reset = 0, 0, 0
adv_err = {a: torch.zeros(specs[a].state_dim, device=dev) for a in specs}
adv_count = {a: 0 for a in specs}
feat_err = {"velocity_commands": 0.0, "gait_phase": 0.0, "gait_phase_cos": 0.0}
rew_err, resets, terminated_total = 0.0, 0, 0
trip_err, trips = 0.0, 0
w_min = torch.full((dims["world"],), float("inf"), device=dev)
w_max = -w_min.clone()
standing_seen = walking_seen = 0
for step in range(args.steps):
    prev_xi = env.sim_context().clone()
    prev_g = {a: env.goal_state(a) for a in specs}
    prev_vel_left, prev_arm_left = vel.time_left.clone(), arm.time_left.clone()
    prev_len = env.episode_length_buf.clone()
    actions = {a: args.action_std * torch.randn(n, env.cfg.action_spaces[a], device=dev) for a in env.cfg.possible_agents}
    obs, rew, terminated, truncated, _ = env.step(actions)
    reset = env.episode_length_buf != prev_len + 1
    resets += int(reset.sum())
    term = terminated["legs"].float()
    terminated_total += int(term.sum())
    # world vs critic
    w, c = env.world_state(), env._obs_buf["critic"]
    w_min, w_max = torch.minimum(w_min, w.amin(0)), torch.maximum(w_max, w.amax(0))
    for k in SHARED:
        shared_err[k] = max(shared_err[k], (w[:, world[k]] - c[:, critic[k]]).abs().max().item())
    # sim context
    changed = (env.sim_context() != prev_xi).any(dim=1)
    xi_bad += int((changed & ~reset).sum())
    xi_reset_changed += int((changed & reset).sum())
    xi_reset += int(reset.sum())
    # goals
    same_cmd = (vel.vel_command_b == prev_g["legs"][:, 0:3]).all(dim=1)
    steady = ~reset & (vel.time_left < prev_vel_left) & (arm.time_left < prev_arm_left) & same_cmd
    for a, spec in specs.items():
        now = env.goal_state(a)
        if steady.any():
            diff = (goals.advance_goal(spec, prev_g[a]) - now)[steady].abs().amax(0)
            adv_err[a] = torch.maximum(adv_err[a], diff)
            adv_count[a] += int(steady.sum())
    f = goals.goal_features(specs["legs"], env.goal_state("legs"))
    o = obs["legs"]
    feat_err["velocity_commands"] = max(feat_err["velocity_commands"], (f[:, 3:6] - o[:, legs_obs["velocity_commands"]]).abs().max().item())
    feat_err["gait_phase"] = max(feat_err["gait_phase"], (f[:, 6:8] - o[:, legs_obs["gait_phase"]]).abs().max().item())
    feat_err["gait_phase_cos"] = max(feat_err["gait_phase_cos"], (f[:, 8:10] - o[:, legs_obs["gait_phase_cos"]]).abs().max().item())
    standing = torch.norm(f[:, 3:6], dim=1) < env.cfg.zero_cmd_eps
    standing_seen += int(standing.sum())
    walking_seen += int((~standing).sum())
    # rewards
    for a in env.cfg.possible_agents:
        rew_err = max(rew_err, (env.task_rewards[a] + env.cfg.termination_penalty * term - rew[a]).abs().max().item())
        floored = env.reward_managers[a]._reward_buf.clamp(min=0.0)
        trip_err = max(trip_err, (env.task_rewards[a] - env.cfg.golem_estop_penalty * env.trip_events[a].float()
                                  - floored).abs().max().item())
        trips += int(env.trip_events[a].sum())

print(f"after {args.steps} steps: {resets} resets ({terminated_total} terminated)")
print("world vs critic, max abs diff: " + ", ".join(f"{k} {v:.1e}" for k, v in shared_err.items()))
check(all(v == 0.0 for v in shared_err.values()), "world's shared terms == critic's exactly")
print("world per term [min, max] over the run:")
for name, s in world.items():
    print(f"  {name:20s} [{w_min[s].min():+9.3f}, {w_max[s].max():+9.3f}]")
check(bool(torch.isfinite(w_min).all() and torch.isfinite(w_max).all()), "world state finite")
print(f"sim_context: changed in {xi_bad} non-reset env-steps; in {xi_reset_changed} of {xi_reset} resets")
check(xi_bad == 0, "sim_context changes only at resets")
for a, spec in specs.items():
    parts, start = [], 0
    for name, size in spec.state_layout.items():
        parts.append(f"{name} {adv_err[a][start:start + size].max().item():.1e}")
        start += size
    print(f"advance_goal({a}) vs next goal_state over {adv_count[a]} env-steps, max abs diff: " + ", ".join(parts))
check(all(adv_err[a].max().item() < 1e-4 for a in specs) and all(adv_count[a] > 0 for a in specs),
      "advance_goal matches the env (< 1e-4)")
print(f"goal_features vs legs observation ({standing_seen} standing / {walking_seen} walking env-steps): "
      + ", ".join(f"{k} {v:.1e}" for k, v in feat_err.items()))
check(all(v == 0.0 for v in feat_err.values()), "goal_features reproduce the observed command and gait phase exactly")
print(f"task_rewards + termination_penalty * terminated vs reward: max abs diff {rew_err:.1e}")
check(rew_err == 0.0, "task_rewards exact")
print(f"task_rewards - golem_estop_penalty * trip_events vs the floored manager reward: max abs diff {trip_err:.1e}"
      f" ({trips} trip starts)")
check(trip_err < 1e-6, "trip_events: task_rewards = floored manager reward + penalty x trip (float rounding)")
print(f"\n{args.task}: {'ALL OK' if not failures else 'FAILURES: ' + '; '.join(failures)}")
sys.stdout.flush()
os._exit(1 if failures else 0)
