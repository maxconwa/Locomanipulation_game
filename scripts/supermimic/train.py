"""Train the LocoManip agents with SuperMimic (arXiv 2610.28480): a world model fit to the simulator's transitions and
SHAC through it, everything from scratch (actors, critics, world model; no checkpoint, no pretraining data).

    python scripts/supermimic/train.py --headless [--task LocoManip-LegsOnly-v0] [--name <run>] [--seed 42]
        [--num_envs 4096] [--max_steps 200160] [--checkpoint_interval 20016] [--agents legs,arms]
        [--warmup_steps 64] [--write_interval 48] [smc.<field>=<value> ...] [env.<field>=... agent.<field>=...]

The loop. After a warm-up of --warmup_steps env steps (the buffer then holds windows; cfg.reward_scale, unless given,
is set from it: the std of the trained agents' per-step task reward without the trip cost), every iteration
  1. collects cfg.collect_steps env steps with the actors' mean actions plus N(0, std^2), std per env from
     ShacTrainer.collection_noise (cfg.action_noise_std, or drawn up to cfg.action_noise_std_max) (untrained agents
     act with zeros), storing per step the pre-step world state, sim context, observations, goal states and
     actor LSTM states, the executed action of every agent, each agent's task reward (env.task_rewards: without the
     fall penalty) and trip flag (env.trip_events; cfg.trip_penalty = the env's golem_estop_penalty), terminated and
     truncated; the normalisers move on real data only, the LSTM states of the envs that
     just ended are zeroed;
  2. takes one world-model update (train_world_model: cfg.wm_batches x cfg.wm_batch_size windows of cfg.window);
  3. takes one SHAC update of every trained actor and its critic (ShacTrainer.update: cfg.imagine_batch imagined
     rollouts of cfg.horizon steps), once the policy has started: for at least cfg.policy_start_iters iterations
     and until the world model's 1-step state error is below cfg.policy_start_rmse (at the latest 5 x
     policy_start_iters) the actors keep their initial weights and the critics wait; then the critics alone are fit
     for cfg.critic_warmup_iters iterations.
cfg.lr decays on a cosine over cfg.lr_decay_iters iterations when set. Agents are trained by task: the legs unless
their action scale is all zero (ArmsOnly), the arms unless actions.arm_pos.residual_gain is 0 (LegsOnly); --agents
overrides it.

The compatibility contract with the PPO pipeline (scripts/skrl/train.py), so that export_marl.py, eval_scorecard.py,
eval_arms.py, progress.sh and plot_progress.py work on these runs unchanged:
- the run is <logs/skrl/locomanip_marl>/<time>_<name>/ with params/{env.yaml, agent.yaml, hydra_overrides.txt,
  task.txt} (+ supermimic.yaml: the resolved SuperMimicCfg and this script's settings); env.<...> / agent.<...>
  arguments are Hydra overrides exactly as there (saved for the evaluation scripts to replay), and an override of a
  field the env cfg's sync_derived writes is refused;
- the actors are recurrent.LSTMGaussianPolicy built from the agent yaml as MarlRunner builds them (with the env cfg's
  policy_output_init_scale and policy_initial_std applied as train.py applies them), one per agent, the untrained
  ones too, so a checkpoint keeps the two-agent layout;
- checkpoints/agent_<env steps>.pt = {agent: {"policy": state_dict, "observation_preprocessor": skrl
  RunningStandardScaler state (float64 running_mean / running_variance / current_count)}} every
  --checkpoint_interval env steps and at the end, beside the env's curriculum/state_<env steps>.pt (state_save_every
  = the checkpoint interval); checkpoints/supermimic_<env steps>.pt holds the world model, critics, optimizers and
  normalisers;
- TensorBoard events in the run folder, on the env-policy-step axis: the env's extras["log"] under skrl's tag names
  (Episode_Reward/legs/..., Metrics/..., Curriculum/...), "Reward / Total reward (mean)" (the agents' summed return
  per finished episode, fall penalty included), "Episode / Total timesteps (mean)", "Reward / Instantaneous reward
  (mean)", plus wm/* (model fit), shac/* (imagination), smc/* (simulated reward, simulated vs imagined H-step return)
  and perf/* (seconds per phase, steps per second, GPU memory).
"""

import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="LocoManip-LegsOnly-v0")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--name", default="supermimic", help="Run directory suffix: <time>_<name>.")
parser.add_argument("--max_steps", type=int, default=200160, help="Env policy steps to train for.")
parser.add_argument("--checkpoint_interval", type=int, default=20016, help="Env policy steps between checkpoints.")
parser.add_argument("--agents", default=None, help="Comma-separated agents to train (default: by task).")
parser.add_argument("--warmup_steps", type=int, default=64, help="Env steps collected before the first update.")
parser.add_argument("--write_interval", type=int, default=48, help="Env steps between TensorBoard points.")
parser.add_argument("--print_seconds", type=float, default=60.0, help="Seconds between progress lines.")
AppLauncher.add_app_launcher_args(parser)
args_cli, rest = parser.parse_known_args()
smc_args = [a for a in rest if a.startswith("smc.")]
hydra_args = [a for a in rest if not a.startswith("smc.")]
sys.argv = [sys.argv[0]] + hydra_args
simulation_app = AppLauncher(args_cli).app

import dataclasses  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import random  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
import traceback  # noqa: E402
from datetime import datetime  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from torch.utils.tensorboard import SummaryWriter  # noqa: E402

from isaaclab.utils.io import dump_yaml  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402

import locomanipulation_game.tasks  # noqa: E402, F401
from locomanipulation_game.supermimic import (  # noqa: E402
    AgentDims, GoalFns, ReplayBuffer, RunningNorm, ShacTrainer, SuperMimicCfg, WorldModel, flatten_lstm, lr_factor,
    set_lr, standard_spec, train_world_model)
from locomanipulation_game.supermimic.buffer import is_transition  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl import goals  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.recurrent import LSTMGaussianPolicy  # noqa: E402
from locomanipulation_game.tasks.locomanip_marl.run_files import cfg_value, env_override_keys  # noqa: E402

# observation terms that are exact copies of goal features: (agent, observation term, feature part, column map
# within the part: observation column i of the term <- feature column map[i] of the part; None = identity)
_XYZW_FROM_WXYZ = [0, 1, 2, 4, 5, 6, 3]  # per arm: (x, y, z, qx, qy, qz, qw) <- (x, y, z, qw, qx, qy, qz)
GOAL_COPIES = [
    ("legs", "velocity_commands", "command_filtered", None),
    ("legs", "gait_phase", "gait_phase_sin", None),
    ("legs", "gait_phase_cos", "gait_phase_cos", None),
    ("arms", "ee_targets", "wrist_targets", _XYZW_FROM_WXYZ + [7 + i for i in _XYZW_FROM_WXYZ]),
    ("arms", "arm_goal", "arm_mode", None),
]


# ---- configuration ---------------------------------------------------------------------------------------------------
def parse_smc_overrides(cfg: SuperMimicCfg, overrides: list[str]) -> SuperMimicCfg:
    """smc.<field>=<yaml value> (smc.<dict field>.<key>=<value> for one entry of a dict field) onto cfg, with the
    field's type kept (yaml lists become tuples for tuple fields)."""
    fields = {f.name: f for f in dataclasses.fields(cfg)}
    for override in overrides:
        key, _, text = override[len("smc."):].partition("=")
        name, _, sub = key.partition(".")
        if name not in fields:
            raise SystemExit(f"[ERROR] {override}: SuperMimicCfg has no field {name!r} (fields: {sorted(fields)})")
        value = yaml.safe_load(text)
        current = getattr(cfg, name)
        if sub:
            if not isinstance(current, dict):
                raise SystemExit(f"[ERROR] {override}: {name} is not a dict field")
            current = dict(current)
            current[sub] = type(current[sub])(value) if sub in current else value
            value = current
        elif isinstance(current, tuple):
            value = tuple(value)
        elif isinstance(current, float):
            value = float(value)
        elif isinstance(current, list) and isinstance(value, str):
            value = [v for v in value.split(",") if v]
        setattr(cfg, name, value)
    return cfg


def default_trained_agents(env_cfg) -> list[str]:
    """An agent is trained iff its actions have an effect: the legs unless their action scale is all zero (ArmsOnly),
    the arms unless the arm action's residual_gain is 0 (LegsOnly)."""
    agents = []
    scale = env_cfg.actions.joint_pos.scale
    values = list(scale.values()) if isinstance(scale, dict) else [scale]
    if any(float(v) != 0.0 for v in values):
        agents.append("legs")
    if float(getattr(env_cfg.actions.arm_pos, "residual_gain", 1.0)) != 0.0:
        agents.append("arms")
    return agents


def term_layout(manager, group: str) -> dict[str, slice]:
    """The observation group's term -> column slice."""
    out, start = {}, 0
    for name, dims in zip(manager.active_terms[group], manager.group_obs_term_dim[group]):
        size = int(np.prod(dims))
        out[name] = slice(start, start + size)
        start += size
    return out


def part_offsets(layout: dict[str, int]) -> dict[str, slice]:
    out, start = {}, 0
    for name, size in layout.items():
        out[name] = slice(start, start + size)
        start += size
    return out


def goal_copy_columns(env, agent: str, spec, obs: torch.Tensor, gstate: torch.Tensor) -> tuple[tuple, tuple]:
    """(observation columns, goal-feature columns) of agent's observation terms that are copies of its goal features
    (GOAL_COPIES, from the observation manager's term layout and goals.py's feature layout), each pair verified exact
    on the given observation; a pair that is not is dropped, said aloud."""
    obs_layout = term_layout(env.observation_manager, agent)
    feat_layout = part_offsets(spec.feature_layout)
    feats = goals.goal_features(spec, gstate)
    obs_cols, feat_cols = [], []
    for who, term, part, colmap in GOAL_COPIES:
        if who != agent or term not in obs_layout or part not in feat_layout:
            continue
        o, f = obs_layout[term], feat_layout[part]
        width = o.stop - o.start
        cmap = list(range(width)) if colmap is None else colmap
        if len(cmap) != width or f.stop - f.start != width:
            print(f"[WARNING] {agent}: {term} ({width}) and goal feature {part} ({f.stop - f.start}) differ in width;"
                  f" not declared a copy")
            continue
        oc = [o.start + i for i in range(width)]
        fc = [f.start + cmap[i] for i in range(width)]
        err = (obs[:, oc] - feats[:, fc]).abs().max().item()
        if err != 0.0:
            print(f"[WARNING] {agent}: observation {term} is not goal feature {part} (max abs diff {err:.3g}); learned")
            continue
        print(f"[INFO] {agent}: observation {term} [{o.start}:{o.stop}] copied from goal feature {part} "
              f"[{f.start}:{f.stop}] (verified exact on {obs.shape[0]} envs)")
        obs_cols += oc
        feat_cols += fc
    return tuple(obs_cols), tuple(feat_cols)


def build_actor(env, agent_cfg: dict, env_cfg, agent: str) -> LSTMGaussianPolicy:
    """agent's actor as recurrent.MarlRunner._generate_models builds it from the agent yaml (and its log std bounds as
    MarlRunner widens them), with scripts/skrl/train.py's initialisation hooks (policy_output_init_scale: the last
    layer's weights scaled, its bias zeroed; policy_initial_std)."""
    policy_cfg, rnn = agent_cfg["models"]["policy"], agent_cfg["models"]["policy"]["rnn"]
    network = policy_cfg["network"][0]
    bounds = {uid: (policy_cfg["min_log_std"], policy_cfg["max_log_std"]) for uid in env.possible_agents}
    per_agent = {**(agent_cfg.get("log_std_bounds") or {}), **getattr(env_cfg, "policy_log_std_bounds", {})}
    bounds.update({uid: (float(low), float(high)) for uid, (low, high) in per_agent.items()})
    actor = LSTMGaussianPolicy(
        observation_space=env.observation_spaces[agent], state_space=env.state_space,
        action_space=env.action_spaces[agent], device=env.device, num_envs=env.num_envs,
        sequence_length=agent_cfg["agent"]["rollouts"], hidden_size=rnn["hidden_size"], num_layers=rnn["num_layers"],
        skip_observations=rnn.get("skip_observations", False), cell_cap=rnn.get("cell_cap", 0.0),
        layers=network["layers"], activation=network["activations"], clip_actions=policy_cfg["clip_actions"],
        clip_log_std=policy_cfg["clip_log_std"], min_log_std=min(low for low, _ in bounds.values()),
        max_log_std=max(high for _, high in bounds.values()), initial_log_std=policy_cfg["initial_log_std"])
    with torch.no_grad():
        scale = env_cfg.policy_output_init_scale.get(agent)
        if scale is not None:
            actor.net[-1].weight.mul_(scale)
            actor.net[-1].bias.zero_()
            print(f"[INFO] {agent}: actor's last layer scaled by {scale} at initialisation, its bias zeroed")
        std = env_cfg.policy_initial_std.get(agent)
        if std is not None:
            actor.log_std_parameter.fill_(math.log(std))
    return actor


def skrl_scaler_state(norm: RunningNorm) -> dict[str, torch.Tensor]:
    """RunningNorm as skrl's RunningStandardScaler state_dict, in its dtype (float64), on the CPU."""
    return {k: v.detach().to("cpu", torch.float64) for k, v in norm.skrl_state().items()}


# ---- logging ---------------------------------------------------------------------------------------------------------
class Tracker:
    """skrl's tracking, on the GPU without a sync per step: per write interval the mean of every one-element entry of
    the env's extras["log"] (tag: the key, or "Info / <key>" without a "/"), the summed reward over agents per step
    (mean, min, max) and per finished episode (return and length, mean, min, max), plus any scalars added."""

    def __init__(self, num_envs: int, device):
        self.device = device
        self.cum_reward = torch.zeros(num_envs, device=device)
        self.cum_steps = torch.zeros(num_envs, device=device)
        self.clear()

    def clear(self):
        self.info: dict[tuple, list] = {}   # key tuple -> [sum vector, steps]
        z = lambda v: torch.tensor(v, device=self.device)  # noqa: E731
        self.step_stats = [z(0.0), z(math.inf), z(-math.inf), 0]          # sum of means, min, max, steps
        self.episodes = [z(0.0), z(0.0), z(0.0), z(math.inf), z(-math.inf), z(math.inf), z(-math.inf)]
        self.scalars: dict[str, list[float]] = {}

    def env_step(self, reward_sum: torch.Tensor, done: torch.Tensor, log: dict):
        keys, values = [], []
        for key, value in log.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    continue
                values.append(value.detach().reshape(()).float())
            else:
                values.append(torch.tensor(float(value), device=self.device))
            keys.append(key if "/" in key else f"Info / {key}")
        if keys:
            vec = torch.stack(values)
            entry = self.info.setdefault(tuple(keys), [torch.zeros_like(vec), 0])
            entry[0] += vec
            entry[1] += 1
        s = self.step_stats
        s[0] += reward_sum.mean()
        s[1] = torch.minimum(s[1], reward_sum.min())
        s[2] = torch.maximum(s[2], reward_sum.max())
        s[3] += 1
        self.cum_reward += reward_sum
        self.cum_steps += 1
        d = done.float()
        e = self.episodes
        e[0] += d.sum()
        e[1] += (self.cum_reward * d).sum()
        e[2] += (self.cum_steps * d).sum()
        inf = torch.full_like(self.cum_reward, math.inf)
        e[3] = torch.minimum(e[3], torch.where(done, self.cum_reward, inf).min())
        e[4] = torch.maximum(e[4], torch.where(done, self.cum_reward, -inf).max())
        e[5] = torch.minimum(e[5], torch.where(done, self.cum_steps, inf).min())
        e[6] = torch.maximum(e[6], torch.where(done, self.cum_steps, -inf).max())
        keep = (~done).float()
        self.cum_reward *= keep
        self.cum_steps *= keep

    def add(self, metrics: dict[str, float]):
        for key, value in metrics.items():
            self.scalars.setdefault(key, []).append(float(value))

    def write(self, writer: SummaryWriter, step: int) -> dict[str, float]:
        out: dict[str, list] = {}
        for keys, (vec, n) in self.info.items():
            for key, value in zip(keys, (vec / max(n, 1)).tolist()):
                out.setdefault(key, []).append((value, n))
        values = {key: sum(v * n for v, n in parts) / sum(n for _, n in parts) for key, parts in out.items()}
        s = self.step_stats
        if s[3]:
            values["Reward / Instantaneous reward (mean)"] = float(s[0]) / s[3]
            values["Reward / Instantaneous reward (min)"] = float(s[1])
            values["Reward / Instantaneous reward (max)"] = float(s[2])
        count, ret_sum, len_sum, rmin, rmax, lmin, lmax = (float(x) for x in self.episodes)
        if count > 0:
            values.update({"Reward / Total reward (mean)": ret_sum / count, "Reward / Total reward (min)": rmin,
                           "Reward / Total reward (max)": rmax, "Episode / Total timesteps (mean)": len_sum / count,
                           "Episode / Total timesteps (min)": lmin, "Episode / Total timesteps (max)": lmax,
                           "smc/episodes_finished": count})
        for key, vals in self.scalars.items():
            values[key] = float(np.mean(vals))
        for key, value in values.items():
            writer.add_scalar(key, value, step)  # nan too, as skrl writes it (the env's metrics before they have data)
        self.clear()
        return values


def recent_windows(buffer: ReplayBuffer, length: int, batch: int) -> dict[str, torch.Tensor] | None:
    """The buffer's newest `length` steps of up to `batch` envs, as ReplayBuffer.sample_windows lays a window out
    (start entry + `length` transitions; alive / next_valid masks): what the current policy has just done, for the
    simulated-vs-imagined return comparison. None while the buffer holds fewer than length + 1 steps."""
    if len(buffer) < length + 1:
        return None
    env = torch.randperm(buffer.num_envs, device=buffer.device)[:batch]
    rows = (buffer.ptr - length - 1 + torch.arange(length + 1, device=buffer.device)) % buffer.capacity
    rows = rows.unsqueeze(0).expand(len(env), -1)
    cols = env.unsqueeze(1)
    out = {}
    for name in buffer.spec:
        out[name] = buffer.data[name][rows[:, :length], cols] if is_transition(name) else buffer.data[name][rows, cols]
    done = (out["terminated"] | out["truncated"]).float()
    out["alive"] = torch.cat([torch.ones_like(done[:, :1]), torch.cumprod(1.0 - done, dim=1)[:, :-1]], dim=1)
    return out


@torch.no_grad()
def compare_returns(trainer: ShacTrainer, buffer: ReplayBuffer, cfg: SuperMimicCfg, batch: int) -> dict[str, float]:
    """The H-step discounted return (no bootstrap; the fall penalty in both) of the newest H real steps against the
    world model's imagination of the current deterministic policy from the same start states, per trained agent; and
    the share still alive after H steps in each."""
    H = cfg.horizon
    win = recent_windows(buffer, H, batch)
    if win is None:
        return {}
    starts = {name: win[name][:, 0] for name in buffer.spec if not is_transition(name)}
    imag = trainer.imagine(starts)
    discount = cfg.gamma ** torch.arange(H, device=buffer.device, dtype=torch.float32)
    terminated = win["terminated"].float()
    last_done = (win["terminated"] | win["truncated"])[:, -1].float()
    out = {"smc/alive_H_sim": float((win["alive"][:, -1] * (1.0 - last_done)).mean()),
           "smc/alive_H_imagined": float(imag.survival[-1].mean())}
    for n in trainer.trained:
        real = ((win[f"reward/{n}"] + cfg.fall_penalty * terminated) * win["alive"] * discount).sum(1)
        imagined = (imag.survival[:-1] * imag.rewards[n] * discount.unsqueeze(1)).sum(0)
        out[f"smc/{n}/return_H_sim"] = float(real.mean())
        out[f"smc/{n}/return_H_imagined"] = float(imagined.mean())
    return out


def process_gpu_mib() -> float:
    """This process's GPU memory as nvidia-smi reports it (MiB; nan if unavailable)."""
    try:
        text = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                              capture_output=True, text=True, timeout=10).stdout
        for line in text.splitlines():
            pid, mem = (x.strip() for x in line.split(","))
            if int(pid) == os.getpid():
                return float(mem)
    except Exception:  # noqa: BLE001
        pass
    return float("nan")


# ---- the run ---------------------------------------------------------------------------------------------------------
@hydra_task_config(args_cli.task, "skrl_mappo_cfg_entry_point")
def main(env_cfg, agent_cfg: dict):
    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.sim.device = args_cli.device
    if args_cli.seed is not None:
        agent_cfg["seed"] = args_cli.seed
    seed = agent_cfg["seed"]
    env_cfg.seed = seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    interval = args_cli.checkpoint_interval
    agent_cfg["agent"]["experiment"]["checkpoint_interval"] = interval
    agent_cfg["trainer"]["timesteps"] = args_cli.max_steps

    cfg = parse_smc_overrides(SuperMimicCfg(), smc_args)
    reward_scale_given = any(a.startswith("smc.reward_scale=") for a in smc_args)
    if args_cli.agents:
        cfg.trained_agents = [a for a in args_cli.agents.split(",") if a]
    elif not cfg.trained_agents:
        cfg.trained_agents = default_trained_agents(env_cfg)
    if not cfg.trained_agents:
        raise SystemExit("[ERROR] no agent to train (--agents)")
    if args_cli.warmup_steps < cfg.window + 1:
        raise SystemExit(f"[ERROR] --warmup_steps {args_cli.warmup_steps}: a window needs {cfg.window + 1} steps")

    log_root = os.path.abspath(os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"]))
    run_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{args_cli.name}"
    agent_cfg["agent"]["experiment"]["directory"] = log_root
    agent_cfg["agent"]["experiment"]["experiment_name"] = run_name
    log_dir = os.path.join(log_root, run_name)
    print(f"[INFO] Logging to {log_dir}")
    overridden = {key: cfg_value(env_cfg, key) for key in env_override_keys(hydra_args)}
    # the curriculum state saved at the checkpoints' steps: an evaluated checkpoint starts from its own step's
    env_cfg.state_save_every = interval
    env_cfg.training = True
    env_cfg.sync_derived()  # after the overrides, so env.yaml records what the run uses
    lost = [key for key, value in overridden.items() if cfg_value(env_cfg, key) != value]
    if lost:
        print(f"[ERROR] These overrides are of fields that follow from others (the env cfg's sync_derived;"
              f" state_save_every from --checkpoint_interval), which would replace them: {lost}."
              f" Set the fields they follow from instead.", flush=True)
        os._exit(2)
    env_cfg.log_dir = log_dir

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped
    device, N = env.device, env.num_envs
    agents = list(env.cfg.possible_agents)
    unknown = set(cfg.trained_agents) - set(agents)
    if unknown:
        raise SystemExit(f"[ERROR] --agents: {sorted(unknown)} are not agents of {args_cli.task} ({agents})")
    cfg.trained_agents = [a for a in agents if a in cfg.trained_agents]
    cfg.fall_penalty = float(env.cfg.termination_penalty)
    cfg.trip_penalty = float(env.cfg.golem_estop_penalty)
    trained = cfg.trained_agents
    print(f"[INFO] SuperMimic on {args_cli.task}: {N} envs, trained agents {trained}, others act with zeros")

    obs, _ = env.reset()
    obs = {a: obs[a].clone() for a in agents}
    specs = {a: goals.goal_spec(env, a) for a in agents}
    goal_fns = {a: GoalFns(features=(lambda g, s=specs[a]: goals.goal_features(s, g)),
                           advance=(lambda g, s=specs[a]: goals.advance_goal(s, g))) for a in agents}
    gstate = {a: env.goal_state(a) for a in agents}
    copies = {a: goal_copy_columns(env, a, specs[a], obs[a], gstate[a]) for a in agents}
    dims = {a: AgentDims(obs_dim=env.cfg.observation_spaces[a], goal_dim=specs[a].feature_dim,
                         action_dim=env.cfg.action_spaces[a], goal_obs_columns=copies[a][0],
                         goal_feature_columns=copies[a][1]) for a in agents}
    S, C = env.world_state().shape[1], env.sim_context().shape[1]
    A = sum(d.action_dim for d in dims.values())
    print(f"[INFO] world state S={S}, sim context C={C}, actions A={A}; " + "; ".join(
        f"{a}: obs {d.obs_dim}, goal state {specs[a].state_dim} -> features {d.goal_dim}" for a, d in dims.items()))

    actors = {a: build_actor(env, agent_cfg, env_cfg, a) for a in agents}
    obs_norms = {a: RunningNorm(dims[a].obs_dim, std_floor=0.0).to(device) for a in agents}
    wm = WorldModel(S, C, dims, cfg).to(device)
    wm_opt = torch.optim.Adam(wm.parameters(), lr=cfg.lr)
    obs_noise = {a: env.obs_noise_scale(a) for a in trained}
    trainer = ShacTrainer(wm, {a: actors[a] for a in trained}, goal_fns, obs_norms, obs_noise, cfg,
                          action_clip=float(env.cfg.clip_actions))
    lstm_sizes = {a: (actors[a].num_layers, actors[a].hidden_size) for a in trained}
    spec = standard_spec(S, C, A, {a: dims[a].obs_dim for a in agents}, {a: specs[a].state_dim for a in agents},
                         lstm_sizes, trips=bool(cfg.trip_penalty))
    buffer = ReplayBuffer(spec, N, cfg.buffer_steps, device=device)
    print(f"[INFO] replay buffer: {cfg.buffer_steps} steps x {N} envs, {buffer.nbytes() / 2**30:.2f} GiB")
    print(f"[INFO] world model {sum(p.numel() for p in wm.parameters()) / 1e6:.2f} M parameters; actors "
          + ", ".join(f"{a} {sum(p.numel() for p in actors[a].parameters()) / 1e3:.0f} k" for a in agents))

    # -- the run's files
    os.makedirs(os.path.join(log_dir, "checkpoints"), exist_ok=True)
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    with open(os.path.join(log_dir, "params", "hydra_overrides.txt"), "w") as f:
        f.writelines(f"{arg}\n" for arg in hydra_args)
    with open(os.path.join(log_dir, "params", "task.txt"), "w") as f:
        f.write(args_cli.task + "\n")

    def write_params():
        resolved = {k: (list(v) if isinstance(v, tuple) else v) for k, v in dataclasses.asdict(cfg).items()}
        script = {k: getattr(args_cli, k) for k in ("task", "num_envs", "seed", "name", "max_steps",
                                                    "checkpoint_interval", "agents", "warmup_steps",
                                                    "write_interval")}
        script.update(seed=seed, num_envs=N, smc_overrides=smc_args,
                      goal_obs_copies={a: {"obs": list(d.goal_obs_columns), "features": list(d.goal_feature_columns)}
                                       for a, d in dims.items()})
        with open(os.path.join(log_dir, "params", "supermimic.yaml"), "w") as f:
            yaml.safe_dump({"supermimic": resolved, "script": script}, f, sort_keys=False)

    write_params()
    writer = SummaryWriter(log_dir=log_dir)
    tracker = Tracker(N, device)

    def save_checkpoint(step: int):
        torch.save({a: {"policy": {k: v.detach().cpu() for k, v in actors[a].state_dict().items()},
                        "observation_preprocessor": skrl_scaler_state(obs_norms[a])} for a in agents},
                   os.path.join(log_dir, "checkpoints", f"agent_{step}.pt"))
        torch.save({"env_steps": step, "iteration": iteration, "cfg": dataclasses.asdict(cfg),
                    "world_model": wm.state_dict(), "wm_optimizer": wm_opt.state_dict(),
                    "critics": {a: trainer.critics[a].state_dict() for a in trained},
                    "critic_optimizers": {a: trainer.critic_opts[a].state_dict() for a in trained},
                    "actor_optimizers": {a: trainer.actor_opts[a].state_dict() for a in trained},
                    "obs_norms": {a: obs_norms[a].state_dict() for a in agents}},
                   os.path.join(log_dir, "checkpoints", f"supermimic_{step}.pt"))
        print(f"[INFO] saved checkpoints/agent_{step}.pt and supermimic_{step}.pt", flush=True)

    copy_err = {a: torch.zeros((), device=device) for a in agents if dims[a].goal_obs_columns}
    lstm = trainer.initial_lstm(N)
    noise_std = trainer.collection_noise(N)   # each env's, for the whole run
    iteration, saved_at = 0, -1

    def collect(steps: int):
        """steps env steps into the buffer with the current actors (noise on). Saves a checkpoint at every multiple
        of the interval (the actors that took those steps; the env saves its curriculum state at the same step)."""
        nonlocal obs, lstm, saved_at
        for _ in range(steps):
            step = {"state": env.world_state().clone(), "ctx": env.sim_context().clone()}
            for a in agents:
                obs_norms[a].update(obs[a])
                step[f"obs/{a}"] = obs[a]
                g = env.goal_state(a)
                step[f"goal/{a}"] = g
                if a in copy_err:
                    d = dims[a]
                    feats = goal_fns[a].features(g)
                    err = obs[a][:, list(d.goal_obs_columns)] - feats[:, list(d.goal_feature_columns)]
                    copy_err[a] = torch.maximum(copy_err[a], err.abs().max())
            for a in trained:
                step[f"h/{a}"], step[f"c/{a}"] = flatten_lstm(lstm[a][0]), flatten_lstm(lstm[a][1])
            action, per_agent, lstm_next = trainer.act(obs, lstm, explore=True, noise_std=noise_std)
            sync()
            t_env = time.time()
            next_obs, rewards, terminated, truncated, extras = env.step(per_agent)
            sync()
            timing["env"] += time.time() - t_env
            timing["env_steps"] += 1
            term, trunc = terminated[agents[0]], truncated[agents[0]]
            step.update({"action": action, "terminated": term, "truncated": trunc})
            for a in agents:
                step[f"reward/{a}"] = env.task_rewards[a].clone()
                if cfg.trip_penalty:
                    step[f"trip/{a}"] = env.trip_events[a].float()
            wm.update_norms(step, goal_fns)
            buffer.add(step)
            done = term | trunc
            lstm = trainer.reset_lstm(lstm_next, done)
            obs = {a: next_obs[a].clone() for a in agents}
            tracker.env_step(sum(rewards[a] for a in agents), done, extras.get("log", {}))
            for a in trained:
                add_tensor(f"smc/{a}/sim_step_reward", env.task_rewards[a].mean())
            env_steps = env.common_step_counter
            if env_steps % interval == 0 and env_steps != saved_at:
                save_checkpoint(env_steps)
                saved_at = env_steps
            if env_steps % args_cli.write_interval == 0:
                write_point(env_steps)

    sums_t: dict[str, list] = {}

    def add_tensor(key, value):
        sums_t.setdefault(key, []).append(value.detach())

    timing = {"collect": 0.0, "model": 0.0, "policy": 0.0, "steps": 0, "iters": 0, "env": 0.0, "env_steps": 0}
    last_print = [time.time(), 0]
    latest: dict[str, float] = {}

    def write_point(step: int):
        for key, vals in sums_t.items():
            tracker.add({key: float(torch.stack(vals).mean())})
        sums_t.clear()
        if timing["iters"]:  # the iterations completed since the last point
            iters = timing["iters"]
            tracker.add({"perf/collect_s_per_iter": timing["collect"] / iters,
                         "perf/model_s_per_iter": timing["model"] / iters,
                         "perf/policy_s_per_iter": timing["policy"] / iters,
                         "perf/env_steps_per_s": timing["steps"] / max(timing["collect"] + timing["model"]
                                                                       + timing["policy"], 1e-9)})
            timing.update(collect=0.0, model=0.0, policy=0.0, steps=0, iters=0)
        if timing["env_steps"]:  # the simulator's share of collecting: env.step alone, per env step
            tracker.add({"perf/env_step_ms": 1e3 * timing["env"] / timing["env_steps"]})
            timing.update(env=0.0, env_steps=0)
        for a, err in copy_err.items():
            tracker.add({f"check/{a}/goal_obs_copy_max_err": float(err)})
        tracker.add({"perf/torch_mem_alloc_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                     "perf/torch_mem_reserved_gib": torch.cuda.memory_reserved(device) / 2**30})
        latest.update(tracker.write(writer, step))

    def progress_line(step: int, force: bool = False):
        now = time.time()
        if not force and now - last_print[0] < args_cli.print_seconds:
            return
        rate = (step - last_print[1]) / max(now - last_print[0], 1e-9)
        last_print[0], last_print[1] = now, step
        k = cfg.window
        g = lambda key, fmt=".3g": format(latest[key], fmt) if key in latest else "-"  # noqa: E731
        parts = [f"it {iteration}", f"steps {step}", f"{rate:.0f} steps/s",
                 f"wm s1 {g('wm/state_rmse_1step')} s{k} {g(f'wm/state_rmse_{k}step')} r {g('wm/reward_rmse')}"
                 f" d {g('wm/termination_accuracy')}"]
        for a in trained:
            parts.append(f"{a} r/step {g(f'smc/{a}/sim_step_reward')} J {g(f'shac/{a}/imagined_return')}"
                         f" H-ret sim {g(f'smc/{a}/return_H_sim')} imag {g(f'smc/{a}/return_H_imagined')}")
        parts.append(f"ep len {g('Episode / Total timesteps (mean)', '.0f')}")
        parts.append(f"s/it col {g('perf/collect_s_per_iter', '.2f')} wm {g('perf/model_s_per_iter', '.2f')}"
                     f" pol {g('perf/policy_s_per_iter', '.2f')}")
        print("[SMC] " + " | ".join(parts), flush=True)

    def sync():
        if torch.cuda.is_available():
            torch.cuda.synchronize(device)

    # -- warm-up: enough data for windows, and the reward scale
    start_time = time.time()
    warmup = min(args_cli.warmup_steps, args_cli.max_steps)
    collect(warmup)
    if not reward_scale_given:
        rewards = torch.cat([wm.trip_free(a, buffer.data[f"reward/{a}"][:len(buffer)], {
            k: v[:len(buffer)] for k, v in buffer.data.items() if k.startswith("trip/")}).reshape(-1) for a in trained])
        cfg.reward_scale = max(float(rewards.std()), 1e-3)
        print(f"[INFO] reward_scale = {cfg.reward_scale:.4g} (std of the trained agents' per-step task reward without"
              f" the trip cost over the {len(buffer)}-step warm-up; mean {float(rewards.mean()):.4g})")
        write_params()
    print(f"[INFO] warm-up: {warmup} steps in {time.time() - start_time:.1f} s", flush=True)

    # -- the loop
    policy_iter = -1   # iterations since the policy started (-1: not yet)
    while env.common_step_counter < args_cli.max_steps:
        factor = lr_factor(cfg, iteration)
        set_lr(wm_opt, cfg.lr * factor)
        trainer.set_lr(factor)
        sync()
        t0 = time.time()
        steps = min(cfg.collect_steps, args_cli.max_steps - env.common_step_counter)
        collect(steps)
        sync()
        t1 = time.time()
        if env.common_step_counter >= args_cli.max_steps:
            timing["collect"] += t1 - t0
            break
        wm_metrics = train_world_model(wm, wm_opt, buffer, goal_fns, cfg)
        sync()
        t2 = time.time()
        if policy_iter < 0 and iteration >= cfg.policy_start_iters and (
                wm_metrics["wm/state_rmse_1step"] < cfg.policy_start_rmse or iteration >= 5 * cfg.policy_start_iters):
            policy_iter = 0
            print(f"[INFO] the policy starts at iteration {iteration} ({env.common_step_counter} env steps; world"
                  f" model 1-step state rmse {wm_metrics['wm/state_rmse_1step']:.3f}); critics alone for"
                  f" {cfg.critic_warmup_iters} iterations", flush=True)
        shac_metrics = {}
        if policy_iter >= 0:
            for _ in range(max(1, cfg.policy_updates)):  # the last update's metrics are the ones logged
                shac_metrics = trainer.update(buffer, step_actors=policy_iter >= cfg.critic_warmup_iters)
            policy_iter += 1
        shac_metrics["smc/policy_iters"] = max(policy_iter, 0)
        sync()
        t3 = time.time()
        timing["collect"] += t1 - t0
        timing["model"] += t2 - t1
        timing["policy"] += t3 - t2
        timing["steps"] += steps
        timing["iters"] += 1
        iteration += 1
        tracker.add({**wm_metrics, **shac_metrics, "smc/lr_factor": factor, "smc/reward_scale": cfg.reward_scale})
        if iteration % max(1, args_cli.write_interval // max(cfg.collect_steps, 1)) == 0:
            tracker.add(compare_returns(trainer, buffer, cfg, min(cfg.imagine_batch, N)))
        progress_line(env.common_step_counter)

    final = env.common_step_counter
    write_point(final)
    if final != saved_at:
        save_checkpoint(final)
        env.save_state()
        print(f"[INFO] saved the last checkpoint, agent_{final}.pt, and its curriculum state")
    progress_line(final, force=True)
    print(f"[INFO] Training time: {time.time() - start_time:.1f} s; {iteration} iterations; peak torch memory"
          f" {torch.cuda.max_memory_allocated(device) / 2**30:.2f} GiB; process GPU memory {process_gpu_mib():.0f} MiB",
          flush=True)
    writer.close()


if __name__ == "__main__":
    try:
        main()
    except SystemExit as stop:  # this script's refusals (SystemExit(message)); a bare exit too
        print(stop, flush=True)
        os._exit(stop.code if isinstance(stop.code, int) else 2)
    except BaseException:  # noqa: BLE001  (Isaac's shutdown hangs on an exception: leave at once)
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()
    os._exit(0)  # Isaac's shutdown hangs otherwise
