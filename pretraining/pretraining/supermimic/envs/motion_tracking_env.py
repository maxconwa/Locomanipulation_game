# Ported from SuperMimic (HybridRobotics, commit 7e94b07, Apache-2.0), src/supermimic/envs/motion_tracking_env.py.
# Changes: imports only.
"""Motion-tracking world model environment."""

from __future__ import annotations

from tensordict import TensorDict

from pretraining.supermimic.envs.world_model_env import WorldModelEnv


class MotionTrackingEnv(WorldModelEnv):
    """WM env for a single-actor policy. Actor obs comes from the reference clip."""

    def get_observations(self) -> TensorDict:
        actor = self.representation.build_actor_obs(self._sensor_history(), self._actor_goal_obs)
        return TensorDict(
            {"actor": actor, "critic": self._critic_obs()},
            batch_size=[self.num_envs],
            device=self.device,
        )
