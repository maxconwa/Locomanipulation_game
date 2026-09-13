from isaaclab.envs import ManagerBasedRLEnv
import torch
class PositiveRewardRLEnv(ManagerBasedRLEnv):
    """Clips the summed per-step reward at zero.

    Equivalent to legged_gym's `only_positive_rewards = True`. Without it, a
    policy facing large early penalties learns that terminating quickly beats
    accumulating negative reward, so it falls over on purpose instead of
    learning to walk.

    Clips the TOTAL, not individual terms: the relative weighting between terms
    still shapes behaviour whenever the sum is positive. Per-term TensorBoard
    logging is unaffected, since the reward manager logs before this clamp.
    """

    def step(self, action: torch.Tensor):
        obs, reward, terminated, truncated, extras = super().step(action)
        clipped = torch.clamp(reward, min=0.0)
        self._n = getattr(self, "_n", 0) + 1
        # if self._n % 20 == 0:
            # print(f"[clip] call {self._n}: raw {reward.mean().item():+.4f} -> {clipped.mean().item():+.4f}")
        self.reward_buf = clipped
        return obs, clipped, terminated, truncated, extras