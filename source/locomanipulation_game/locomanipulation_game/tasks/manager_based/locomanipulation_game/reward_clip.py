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

class NegativeRewardRLEnv(ManagerBasedRLEnv):
    def step(self, action: torch.Tensor):
            obs, reward, terminated, truncated, extras = super().step(action)
            step_rew = self.reward_manager._step_reward
            nan_mask = torch.isnan(step_rew)
            if nan_mask.any():
                names = self.reward_manager._term_names
                env_ids = nan_mask.any(dim=1).nonzero(as_tuple=True)[0]
                bad_terms = [names[i] for i in nan_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()]
                hits_z = self.scene["height_scanner"].data.ray_hits_w[env_ids, :, 2]
                print(f"[nan] step {self.common_step_counter}  terms: {bad_terms}")
                print(f"[nan] env ids:    {env_ids.tolist()}")
                print(f"[nan] terminated: {terminated[env_ids].tolist()}")
                print(f"[nan] truncated:  {truncated[env_ids].tolist()}")
                print(f"[nan] root_pos_w: {self.scene['robot'].data.root_pos_w[env_ids].tolist()}")
                print(f"[nan] inf rays:   {torch.isinf(hits_z).sum(dim=1).tolist()} of {hits_z.shape[1]}")
                raise RuntimeError("NaN reward term, see [nan] lines above")
            clipped = torch.clamp(reward, max=0.0)
            self._n = getattr(self, "_n", 0) + 1
            # if self._n % 20 == 0:
                # print(f"[clip] call {self._n}: raw {reward.mean().item():+.4f} -> {clipped.mean().item():+.4f}")
            self.reward_buf = clipped
            return obs, clipped, terminated, truncated, extras
