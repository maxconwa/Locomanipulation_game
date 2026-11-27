"""SteerableArms on LocoManip-ArmsOnly-v0 in Isaac Lab: the arms stack (IK trajectory, gravity offset, pose-error
integral, freeze, the policy's residual) driven by commands written straight into the arm command, never through its
resampler (which draws random goals and redraws on collision).

A goal (interface.py) does what the command does at a goal event, for the given poses: the command (believed pose)
set, the seed posture solved (nearest table row, then its seed_solve), the reach bookkeeping restarted; the action's
trajectory then moves its target to the new seed, and the freeze releases (it is per env, so a goal on either arm
releases both). A nudge sets only the command: the posture stays, the integral and the residual
absorb the change.

Import after the AppLauncher has started (it imports Isaac Lab). The env is reset once, at the first reset(); later
reset() calls carry on from the current state (the benchmark's schedules always start with a goal).
"""

import numpy as np
import torch
from locomanipulation_game.tasks.locomanip_marl.mdp.commands import _relative

from .interface import StepLog


class IsaacArms:
    def __init__(self, env, runner, zero_actions: bool):
        """env: the SkrlVecEnvWrapper; runner: the MarlRunner with the checkpoint loaded, in eval mode; zero_actions:
        the arms' actions zero (the controller alone, as evals/isaac/arms/get_scorecard.py --ik_only), else the policy's mean action."""
        self.env, self.base, self.runner = env, env.unwrapped, runner
        self.arm = self.base.command_manager.get_term("arm_targets")
        self.action = self.base.action_manager.get_term("arm_pos")
        self.robot = self.base.scene["robot"]
        self.zero_actions = zero_actions
        self.num_envs, self.num_arms, self.dt = self.base.num_envs, len(self.arm.body_ids), self.base.step_dt
        self.device = self.base.device
        self._obs = self._states = None
        self.arm_joints = torch.cat([torch.as_tensor(ids) for ids in self.arm._arm_joint_ids]).to(self.device)
        self.arm.goals_enabled, self.arm.goals_enabled_at = True, None

    def _achieved(self) -> torch.Tensor:
        body = torch.cat([self.robot.data.body_pos_w[:, self.arm.body_ids],
                          self.robot.data.body_quat_w[:, self.arm.body_ids]], dim=-1)
        return _relative(self.robot.data.root_pos_w, self.robot.data.root_quat_w, body)

    def _log(self) -> StepLog:
        """The achieved poses, and the stack's state as torch tensors (copied to numpy only where the runner keeps
        them: a copy per step would sync the GPU five more times a step)."""
        extras = {"joint_pos": self.robot.data.joint_pos[:, self.arm_joints].clone(),
                  "residual": self.action._residual.clone(), "frozen": self.action.frozen.clone(),
                  "seed_q": self.arm.seed_q.flatten(1).clone()}
        if hasattr(self.action, "_integral"):
            extras["integral"] = self.action._integral.clone()
        return StepLog(self._achieved().cpu().numpy().astype(np.float64), extras)

    def reset(self) -> StepLog:
        if self._obs is None:
            with torch.inference_mode():
                self._obs, _ = self.base.reset()
                self._states = self.base.state()
        return self._log()

    @torch.inference_mode()
    def set_command(self, poses: np.ndarray, goal: np.ndarray) -> None:
        arm = self.arm
        p = torch.as_tensor(poses, device=self.device, dtype=torch.float32)
        g = torch.as_tensor(goal, device=self.device, dtype=torch.bool)
        arm.believed_b[:] = p
        envs = g.any(dim=1)
        if envs.any():
            ids = envs.nonzero().squeeze(1)
            seed = arm.nearest_joints(p[ids])
            if arm.cfg.seed_solve:
                seed = arm._kin.solve(p[ids], seed, arm._solve_low, arm._solve_high, iters=arm.cfg.seed_solve_iters,
                                      damping=arm.cfg.seed_solve_damping, rot_weight=arm.cfg.seed_solve_rot_weight)
            arm.seed_q[ids] = torch.where(g[ids].unsqueeze(-1), seed, arm.seed_q[ids])
            arm.arm_mode[ids] = True
            arm.goal_reached[ids] = False
            arm.hold_time[ids] = 0.0
            arm.goal_best_error[ids] = float("inf")
            arm.goal_best_rot_error[ids] = float("inf")
        arm.invalidate_errors()

    @torch.inference_mode()
    def step(self) -> StepLog:
        outputs = self.runner.agent.act(self._obs, {a: self._states for a in self.base.possible_agents},
                                        timestep=0, timesteps=0)
        actions = {a: outputs[-1][a].get("mean_actions", outputs[0][a]) for a in self.base.possible_agents}
        if self.zero_actions:
            actions["arms"] = torch.zeros_like(actions["arms"])
        self._obs, _, terminated, truncated, _ = self.base.step(actions)
        self._states = self.base.state()
        self.runner.agent.reset_rnn(terminated["arms"] | truncated["arms"])  # a reset env starts from a zero state
        log = self._log()
        log.extras["reset"] = terminated["arms"] | truncated["arms"]
        return log
