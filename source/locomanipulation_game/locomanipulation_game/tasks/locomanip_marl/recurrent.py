"""Recurrent actors, as ALMI-Open's (rsl_rl RNNActorMLPCritic): an LSTM over each actor's observations ahead of its MLP,
optionally with the raw observations concatenated to the LSTM's output (skip_observations); the critics stay MLPs.

skrl 2.1 has a recurrent PPO (PPO_RNN) but no recurrent MAPPO, and its yaml model instantiators build no RNNs. Here:
LSTMGaussianPolicy is the actor; RecurrentMAPPO is MAPPO with PPO_RNN's handling of the actors' LSTM state (kept per
env, zeroed at episode ends, stored with each step, and trained on each env's whole rollout as one sequence from its
stored first state); MarlRunner is skrl's Runner building both when the yaml's models.policy has an `rnn` entry.
"""

from __future__ import annotations

import itertools

import torch
import torch.nn as nn
import torch.nn.functional as F
from skrl import config
from skrl.models.torch import GaussianMixin, Model
from skrl.multi_agents.torch import MultiAgent
from skrl.multi_agents.torch.mappo import MAPPO
from skrl.multi_agents.torch.mappo.mappo import compute_gae
from skrl.resources.schedulers.torch import KLAdaptiveLR
from skrl.utils.runner.torch import Runner

__all__ = ["LSTMGaussianPolicy", "MarlRunner", "RecurrentMAPPO"]

_ACTIVATIONS = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}


class LSTMGaussianPolicy(GaussianMixin, Model):
    """observations -> LSTM -> MLP (layers, activation) -> mean actions, with a state-independent log std parameter.
    With skip_observations the MLP's input is the LSTM's output concatenated with the raw observations. With
    normalize_hidden (one layer only) the LSTM's output h is scaled to unit length after every step, both what the MLP
    sees and the h carried to the next step; the cell state c is left as it is. Such a policy carries a
    `normalize_hidden_flag` buffer in its state dict, which export_marl.py reads to do the same in the exported actor.
    With cell_cap s > 0 the cell state is squashed after every step, c <- s tanh(c / s), so it stays inside (-s, s)
    and cannot grow without bound over a long run (Mid360v2's mean |c| rose from 4 to 11 over a 180 s stand); h keeps
    its usual o tanh(c). Such a policy carries a `cell_cap_value` buffer, which export_marl.py reads.

    Inputs carry the LSTM state as inputs["rnn"] = [h, c]. Without "terminated" in the inputs (acting) the LSTM takes
    one step from it. With them (an update) the observations are sequences of sequence_length steps, the state is each
    sequence's stored first one, and it is zeroed after every step that ended an episode (terminated or truncated).
    """

    def __init__(self, *, observation_space, state_space, action_space, device, num_envs: int, sequence_length: int,
                 hidden_size: int = 64, num_layers: int = 1, skip_observations: bool = False,
                 normalize_hidden: bool = False, cell_cap: float = 0.0,
                 layers: list[int] = (), activation: str = "elu",
                 clip_actions: bool = False, clip_log_std: bool = True, min_log_std: float = -20.0,
                 max_log_std: float = 2.0, initial_log_std: float = 0.0):
        Model.__init__(self, observation_space=observation_space, state_space=state_space,
                       action_space=action_space, device=device)
        GaussianMixin.__init__(self, clip_actions=clip_actions, clip_log_std=clip_log_std,
                               min_log_std=min_log_std, max_log_std=max_log_std)
        self.num_envs, self.sequence_length = num_envs, sequence_length
        self.hidden_size, self.num_layers = hidden_size, num_layers
        self.skip_observations = skip_observations
        self.lstm = nn.LSTM(self.num_observations, hidden_size, num_layers, batch_first=True)
        self.normalize_hidden = normalize_hidden
        if normalize_hidden:
            if num_layers != 1:
                raise ValueError("normalize_hidden needs a one-layer LSTM (its output is then its h)")
            # only in these checkpoints, so older ones load as before; export_marl.py reads it
            self.register_buffer("normalize_hidden_flag", torch.ones(1))
        self.cell_cap = float(cell_cap)
        if self.cell_cap > 0.0:
            self.register_buffer("cell_cap_value", torch.tensor([self.cell_cap]))
        mlp, width = [], hidden_size + (self.num_observations if skip_observations else 0)
        for size in layers:
            mlp += [nn.Linear(width, size), _ACTIVATIONS[activation]()]
            width = size
        self.net = nn.Sequential(*mlp, nn.Linear(width, self.num_actions))
        self.log_std_parameter = nn.Parameter(torch.full((self.num_actions,), float(initial_log_std)))
        self.to(self.device)

    def get_specification(self):
        size = (self.num_layers, self.num_envs, self.hidden_size)
        return {"rnn": {"sequence_length": self.sequence_length, "sizes": [size, size]}}

    def compute(self, inputs, role=""):
        observations = inputs["observations"]
        h, c = inputs["rnn"]
        if "terminated" in inputs:
            length = self.sequence_length
            steps = observations.view(-1, length, observations.shape[-1])
            h = h.view(self.num_layers, -1, length, self.hidden_size)[:, :, 0].contiguous()
            c = c.view(self.num_layers, -1, length, self.hidden_size)[:, :, 0].contiguous()
            keep = (~(inputs["terminated"] | inputs["truncated"])).view(-1, length).float()
            outputs = []
            for t in range(length):
                out, (h, c) = self.lstm(steps[:, t : t + 1], (h, c))
                if self.normalize_hidden:
                    h = F.normalize(h, dim=-1, eps=1e-6)
                    out = h[-1].unsqueeze(1)
                if self.cell_cap > 0.0:
                    c = self.cell_cap * torch.tanh(c / self.cell_cap)
                outputs.append(out)
                mask = keep[:, t].view(1, -1, 1)
                h, c = h * mask, c * mask
            features = torch.cat(outputs, dim=1).reshape(-1, self.hidden_size)
        else:
            out, (h, c) = self.lstm(observations.unsqueeze(1), (h, c))
            if self.normalize_hidden:
                h = F.normalize(h, dim=-1, eps=1e-6)
                out = h[-1].unsqueeze(1)
            if self.cell_cap > 0.0:
                c = self.cell_cap * torch.tanh(c / self.cell_cap)
            features = out.squeeze(1)
        if self.skip_observations:
            features = torch.cat([features, observations], dim=-1)  # both env-major, in step order
        return self.net(features), {"log_std": self.log_std_parameter, "rnn": [h, c]}


class RecurrentMAPPO(MAPPO):
    """skrl's MAPPO with recurrent policies (critics without state), after skrl's PPO_RNN.

    While training, record_transition stores each step's input LSTM state and then moves on to the state the act
    produced, zeroed for envs whose episode ended. Outside training (evaluation scripts, which call act only) act
    moves on itself; reset_rnn(done) zeroes the envs that were reset.
    """

    def init(self, *, trainer_cfg=None) -> None:
        super().init(trainer_cfg=trainer_cfg)
        self._rnn_names, self._rnn_states, self._rnn_next, self._sequence_length = {}, {}, {}, {}
        for uid in self.possible_agents:
            spec = self.policies[uid].get_specification().get("rnn", {})
            self._sequence_length[uid] = spec.get("sequence_length", 1)
            self._rnn_names[uid], self._rnn_states[uid] = [], []
            for i, size in enumerate(spec.get("sizes", [])):
                if self.memories:
                    self.memories[uid].create_tensor(name=f"rnn_policy_{i}", size=(size[0], size[2]),
                                                     dtype=torch.float32, keep_dimensions=True)
                self._rnn_names[uid].append(f"rnn_policy_{i}")
                self._rnn_states[uid].append(torch.zeros(size, dtype=torch.float32, device=self.device))
            if self._rnn_names[uid] and self.memories and self._sequence_length[uid] != self.memories[uid].memory_size:
                raise ValueError(f"{uid}: sequence_length {self._sequence_length[uid]} must be the rollout length "
                                 f"{self.memories[uid].memory_size} (each env's rollout is one sequence)")

    def reset_rnn(self, done: torch.Tensor) -> None:
        """Zero the LSTM state of the envs where done (bool, (num_envs,) or (num_envs, 1))."""
        ids = done.view(-1).nonzero(as_tuple=False)[:, 0]
        if ids.numel():
            for states in self._rnn_states.values():
                for s in states:
                    s[:, ids] = 0.0

    def act(self, observations, states, *, timestep: int, timesteps: int):
        actions, log_prob, outputs, current_values = {}, {}, {}, {}
        for uid in self.possible_agents:
            inputs = {
                "observations": self._observation_preprocessor[uid](observations[uid]),
                "states": self._state_preprocessor[uid](states[uid]),
            }
            policy_inputs = {**inputs, "rnn": self._rnn_states[uid]} if self._rnn_names[uid] else inputs
            with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                actions[uid], outputs[uid] = self.policies[uid].act(policy_inputs, role="policy")
                log_prob[uid] = outputs[uid]["log_prob"]
                if self.training:
                    values, _ = self.values[uid].act(inputs, role="value")
                    current_values[uid] = self._value_preprocessor[uid](values, inverse=True)
            if self._rnn_names[uid]:
                self._rnn_next[uid] = outputs[uid]["rnn"]
                if not self.training:
                    self._rnn_states[uid] = self._rnn_next[uid]
        self._current_log_prob = log_prob
        self._current_values = current_values
        return actions, outputs

    def record_transition(self, *, observations, states, actions, rewards, next_observations, next_states,
                          terminated, truncated, infos, timestep: int, timesteps: int) -> None:
        MultiAgent.record_transition(
            self, observations=observations, states=states, actions=actions, rewards=rewards,
            next_observations=next_observations, next_states=next_states, terminated=terminated,
            truncated=truncated, infos=infos, timestep=timestep, timesteps=timesteps,
        )
        if not self.training:
            return
        self._current_next_observations = next_observations
        self._current_next_states = next_states
        for uid in self.possible_agents:
            if self.cfg.rewards_shaper is not None:
                rewards[uid] = self.cfg.rewards_shaper(rewards[uid], timestep, timesteps)
            if self.cfg.time_limit_bootstrap[uid] and truncated[uid].any():
                with torch.no_grad():
                    inputs = {
                        "observations": self._observation_preprocessor[uid](next_observations[uid]),
                        "states": self._state_preprocessor[uid](next_states[uid]),
                    }
                    next_values, _ = self.values[uid].act(inputs, role="value")
                    next_values = self._value_preprocessor[uid](next_values, inverse=True)
                rewards[uid] += self.cfg.discount_factor[uid] * next_values * truncated[uid]
            rnn = {name: s.transpose(0, 1) for name, s in zip(self._rnn_names[uid], self._rnn_states[uid])}
            self.memories[uid].add_samples(
                observations=observations[uid], states=states[uid], actions=actions[uid], rewards=rewards[uid],
                terminated=terminated[uid], truncated=truncated[uid], log_prob=self._current_log_prob[uid],
                values=self._current_values[uid], **rnn,
            )
            if self._rnn_names[uid]:
                self._rnn_states[uid] = self._rnn_next[uid]
                ids = (terminated[uid] | truncated[uid]).view(-1).nonzero(as_tuple=False)[:, 0]
                if ids.numel():
                    for s in self._rnn_states[uid]:
                        s[:, ids] = 0.0

    def update(self, *, timestep: int, timesteps: int, uid: str) -> None:
        if not self._rnn_names[uid]:
            return super().update(timestep=timestep, timesteps=timesteps, uid=uid)
        policy, value, memory = self.policies[uid], self.values[uid], self.memories[uid]

        with torch.no_grad(), torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
            inputs = {
                "observations": self._observation_preprocessor[uid](self._current_next_observations[uid]),
                "states": self._state_preprocessor[uid](self._current_next_states[uid]),
            }
            value.enable_training_mode(False)
            last_values, _ = value.act(inputs, role="value")
            value.enable_training_mode(True)
            last_values = self._value_preprocessor[uid](last_values, inverse=True)

        values = memory.get_tensor_by_name("values")
        returns, advantages = compute_gae(
            rewards=memory.get_tensor_by_name("rewards"),
            terminated=memory.get_tensor_by_name("terminated"),
            truncated=memory.get_tensor_by_name("truncated"),
            values=values,
            last_values=last_values,
            discount_factor=self.cfg.discount_factor[uid],
            lambda_coefficient=self.cfg.gae_lambda[uid],
            time_limit_bootstrap=self.cfg.time_limit_bootstrap[uid],
        )
        memory.set_tensor_by_name("values", self._value_preprocessor[uid](values, train=True))
        memory.set_tensor_by_name("returns", self._value_preprocessor[uid](returns, train=True))
        memory.set_tensor_by_name("advantages", advantages)

        # whole sequences: each mini-batch is a set of envs, each with its rollout in order
        names = self._tensors_names + ["terminated", "truncated"]
        mini_batches, length = self.cfg.mini_batches[uid], self._sequence_length[uid]
        batches = memory.sample_all(names=names, mini_batches=mini_batches, sequence_length=length)
        rnn_batches = memory.sample_all(names=self._rnn_names[uid], mini_batches=mini_batches, sequence_length=length)

        cumulative_policy_loss = cumulative_entropy_loss = cumulative_value_loss = 0
        for epoch in range(self.cfg.learning_epochs[uid]):
            kl_divergences = []
            for batch, rnn_batch in zip(batches, rnn_batches):
                (sampled_observations, sampled_states, sampled_actions, sampled_log_prob, sampled_values,
                 sampled_returns, sampled_advantages, sampled_terminated, sampled_truncated) = batch
                with torch.autocast(device_type=self._device_type, enabled=self.cfg.mixed_precision):
                    inputs = {
                        "observations": self._observation_preprocessor[uid](sampled_observations, train=not epoch),
                        "states": self._state_preprocessor[uid](sampled_states, train=not epoch),
                    }
                    _, outputs = policy.act({
                        **inputs, "taken_actions": sampled_actions, "rnn": [s.transpose(0, 1) for s in rnn_batch],
                        "terminated": sampled_terminated, "truncated": sampled_truncated,
                    }, role="policy")
                    next_log_prob = outputs["log_prob"]

                    with torch.no_grad():
                        ratio = next_log_prob - sampled_log_prob
                        kl_divergence = ((torch.exp(ratio) - 1) - ratio).mean()
                        kl_divergences.append(kl_divergence)
                    if self.cfg.kl_threshold[uid] and kl_divergence > self.cfg.kl_threshold[uid]:
                        break

                    if self.cfg.entropy_loss_scale[uid]:
                        entropy_loss = -self.cfg.entropy_loss_scale[uid] * policy.get_entropy(role="policy").mean()
                    else:
                        entropy_loss = 0

                    ratio = torch.exp(next_log_prob - sampled_log_prob)
                    surrogate = sampled_advantages * ratio
                    surrogate_clipped = sampled_advantages * torch.clip(
                        ratio, 1.0 - self.cfg.ratio_clip[uid], 1.0 + self.cfg.ratio_clip[uid]
                    )
                    policy_loss = -torch.min(surrogate, surrogate_clipped).mean()

                    predicted_values, _ = value.act(inputs, role="value")
                    if self.cfg.value_clip[uid] > 0:
                        predicted_values = sampled_values + torch.clip(
                            predicted_values - sampled_values, min=-self.cfg.value_clip[uid], max=self.cfg.value_clip[uid]
                        )
                    value_loss = self.cfg.value_loss_scale[uid] * F.mse_loss(sampled_returns, predicted_values)

                self.optimizers[uid].zero_grad()
                self.scaler.scale(policy_loss + entropy_loss + value_loss).backward()
                if config.torch.is_distributed:
                    policy.reduce_parameters()
                    value.reduce_parameters()
                if self.cfg.grad_norm_clip[uid] > 0:
                    self.scaler.unscale_(self.optimizers[uid])
                    nn.utils.clip_grad_norm_(itertools.chain(policy.parameters(), value.parameters()),
                                             self.cfg.grad_norm_clip[uid])
                self.scaler.step(self.optimizers[uid])
                self.scaler.update()

                cumulative_policy_loss += policy_loss.item()
                cumulative_value_loss += value_loss.item()
                if self.cfg.entropy_loss_scale[uid]:
                    cumulative_entropy_loss += entropy_loss.item()

            if self.schedulers[uid]:
                if isinstance(self.schedulers[uid], KLAdaptiveLR):
                    kl = torch.tensor(kl_divergences, device=self.device).mean()
                    if config.torch.is_distributed:
                        torch.distributed.all_reduce(kl, op=torch.distributed.ReduceOp.SUM)
                        kl /= config.torch.world_size
                    self.schedulers[uid].step(kl.item())
                else:
                    self.schedulers[uid].step()

        steps = self.cfg.learning_epochs[uid] * self.cfg.mini_batches[uid]
        self.track_data(f"Loss / Policy loss ({uid})", cumulative_policy_loss / steps)
        self.track_data(f"Loss / Value loss ({uid})", cumulative_value_loss / steps)
        if self.cfg.entropy_loss_scale[uid]:
            self.track_data(f"Loss / Entropy loss ({uid})", cumulative_entropy_loss / steps)
        self.track_data(f"Policy / Standard deviation ({uid})", policy.distribution(role="policy").stddev.mean().item())
        if self.schedulers[uid]:
            self.track_data(f"Learning / Learning rate ({uid})", self.schedulers[uid].get_last_lr()[0])


class MarlRunner(Runner):
    """skrl's Runner; with models.policy.rnn ({hidden_size, num_layers, skip_observations}) in the cfg, the policies are
    LSTMGaussianPolicy (the MLP after the LSTM from models.policy.network[0]) and the agent RecurrentMAPPO."""

    def __init__(self, env, cfg: dict, **kwargs):
        self._rnn_cfg = cfg["models"]["policy"].get("rnn")
        super().__init__(env, cfg, **kwargs)

    def _generate_models(self, env, cfg: dict):
        if not self._rnn_cfg:
            return super()._generate_models(env, cfg)
        policy_cfg = cfg["models"].pop("policy")
        models = super()._generate_models(env, cfg)  # the critics
        network = policy_cfg["network"][0]
        for uid in env.possible_agents:
            models[uid]["policy"] = LSTMGaussianPolicy(
                observation_space=env.observation_spaces[uid], state_space=env.state_spaces[uid],
                action_space=env.action_spaces[uid], device=env.device, num_envs=env.num_envs,
                sequence_length=cfg["agent"]["rollouts"], hidden_size=self._rnn_cfg["hidden_size"],
                num_layers=self._rnn_cfg["num_layers"],
                skip_observations=self._rnn_cfg.get("skip_observations", False),
                normalize_hidden=self._rnn_cfg.get("normalize_hidden", False),
                cell_cap=self._rnn_cfg.get("cell_cap", 0.0),
                layers=network["layers"], activation=network["activations"],
                clip_actions=policy_cfg["clip_actions"], clip_log_std=policy_cfg["clip_log_std"],
                min_log_std=policy_cfg["min_log_std"], max_log_std=policy_cfg["max_log_std"],
                initial_log_std=policy_cfg["initial_log_std"],
            )
        return models

    def _component(self, name: str):
        if self._rnn_cfg and name.lower() == "mappo":
            return RecurrentMAPPO
        return super()._component(name)
