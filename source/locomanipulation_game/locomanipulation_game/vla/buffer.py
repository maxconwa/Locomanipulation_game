"""The DAgger replay: a ring over time of what each env's policy saw and the teacher's action there.

Per step and env: the two wrist images and the head image's lower part (uint8), the proprioception, the teacher's action, the episode's number, and a
pointer to the VLM's tokens in force (they are refreshed every few steps, so stored once per refresh in their own
ring per env). The big arrays are numpy memmaps under the run's folder. A sample is a step and env with the teacher's
actions of the following steps of the same episode as the chunk's target.

The images are stored every `image_every` steps only (consecutive frames are nearly the same picture, and they are
the buffer's size: 0.45 MB per step and env), and those steps are the ones sampled; the actions of every step are
kept, for the chunks' targets. So the same disk holds image_every times the episodes.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class DaggerBuffer:
    def __init__(self, directory: str, steps: int, num_envs: int, wrist_shape: tuple[int, ...],
                 head_shape: tuple[int, ...], proprio_dim: int,
                 action_dim: int, vlm_shape: tuple[int, int], vlm_refresh: int, horizon: int,
                 image_every: int = 1):
        os.makedirs(directory, exist_ok=True)
        assert steps % image_every == 0
        self.steps, self.num_envs, self.horizon, self.image_every = steps, num_envs, horizon, image_every
        self.image_rows = steps // image_every
        self.vlm_slots = max(8, 2 * steps // vlm_refresh)  # more than the refreshes of `steps` steps
        mm = lambda name, dtype, shape: np.lib.format.open_memmap(  # noqa: E731
            os.path.join(directory, name + ".npy"), mode="w+", dtype=dtype, shape=shape)
        self.wrist = mm("wrist", np.uint8, (self.image_rows, num_envs, *wrist_shape))
        self.head = mm("head", np.uint8, (self.image_rows, num_envs, *head_shape))
        self.blocks = np.zeros((steps, num_envs, 6), dtype=np.float32)
        self.vlm = mm("vlm", np.float16, (self.vlm_slots, num_envs, *vlm_shape))
        self.vlm_mask = np.zeros((self.vlm_slots, num_envs, vlm_shape[0]), dtype=bool)
        self.proprio = np.zeros((steps, num_envs, proprio_dim), dtype=np.float32)
        self.action = np.zeros((steps, num_envs, action_dim), dtype=np.float32)
        self.episode = np.full((steps, num_envs), -1, dtype=np.int64)
        self.vlm_ptr = np.zeros((steps, num_envs), dtype=np.int32)
        self.count = 0  # steps written so far
        self._vlm_count = np.zeros(num_envs, dtype=np.int64)

    def add_vlm(self, env_ids: np.ndarray, tokens: torch.Tensor, mask: torch.Tensor):
        """The VLM's fresh tokens of the envs env_ids: they are the ones in force from the next add() on."""
        slots = self._vlm_count[env_ids] % self.vlm_slots
        self.vlm[slots, env_ids] = tokens.cpu().numpy()
        self.vlm_mask[slots, env_ids] = mask.cpu().numpy()
        self._vlm_count[env_ids] += 1

    def add(self, wrist: torch.Tensor, head: torch.Tensor, proprio: torch.Tensor, action: torch.Tensor,
            blocks: torch.Tensor, episode: np.ndarray):
        """One step of every env: wrist and head images (N, ...) uint8, proprio, the teacher's action, the instructed
        block's and its target's positions (N, 6; privileged, the auxiliary loss's target), the episodes' numbers."""
        t = self.count % self.steps
        if self.count % self.image_every == 0:
            row = (self.count // self.image_every) % self.image_rows
            self.wrist[row] = wrist.cpu().numpy()
            self.head[row] = head.cpu().numpy()
        self.blocks[t] = blocks.cpu().numpy()
        self.proprio[t] = proprio.cpu().numpy()
        self.action[t] = action.cpu().numpy()
        self.episode[t] = episode
        self.vlm_ptr[t] = (self._vlm_count - 1) % self.vlm_slots
        self.count += 1

    def __len__(self) -> int:
        return min(self.count, self.steps) * self.num_envs

    def sample(self, batch: int, device: str) -> dict[str, torch.Tensor]:
        """A batch: the observation of a random stored step and env, the teacher's actions from it on (horizon) and
        which of them belong to its episode (valid)."""
        last = (self.count - 1) // self.image_every  # the newest step with images, in image steps
        t = (last - np.random.randint(0, min(last + 1, self.image_rows), size=batch)) * self.image_every  # absolute step numbers
        e = np.random.randint(0, self.num_envs, size=batch)
        ahead = t[:, None] + np.arange(self.horizon)[None]
        inside = ahead < self.count
        rows = ahead % self.steps
        valid = inside & (self.episode[rows, e[:, None]] == self.episode[t % self.steps, e][:, None])
        target = self.action[rows, e[:, None]]
        i = t % self.steps
        ptr = self.vlm_ptr[i, e]
        image = (t // self.image_every) % self.image_rows
        order = np.argsort(image)  # the memmap reads in file order
        wrist = np.empty((batch, *self.wrist.shape[2:]), dtype=np.uint8)
        wrist[order] = self.wrist[image[order], e[order]]
        head = np.empty((batch, *self.head.shape[2:]), dtype=np.uint8)
        head[order] = self.head[image[order], e[order]]
        to = lambda x: torch.from_numpy(x).to(device, non_blocking=True)  # noqa: E731
        return {"wrist": to(wrist), "head": to(head), "blocks": to(self.blocks[i, e]), "vlm": to(self.vlm[ptr, e]), "vlm_mask": to(self.vlm_mask[ptr, e]),
                "proprio": to(self.proprio[i, e]), "target": to(target), "valid": to(valid)}
