"""Runs the policy on the env's cameras as the robot would: the VLM every few steps, the action head every few steps
for a chunk whose next actions are executed until the next query."""

from __future__ import annotations

import torch

from .encoders import VLM_DIM, VLM_TOKENS, WORD_TOKENS, FrozenEncoders
from .model import ACTION_DIM, VlaPolicy


class PolicyRunner:
    def __init__(self, env, encoders: FrozenEncoders, policy: VlaPolicy, replan: int = 4, vlm_every: int = 8):
        self.env, self.encoders, self.policy = env, encoders, policy
        self.replan, self.vlm_every = replan, vlm_every
        n, dev = env.num_envs, env.device
        # the VLM's tokens, then the instruction's word embeddings
        self.vlm_tokens = torch.zeros(n, VLM_TOKENS + WORD_TOKENS, VLM_DIM, dtype=torch.float16, device=dev)
        self.vlm_mask = torch.zeros(n, VLM_TOKENS + WORD_TOKENS, dtype=torch.bool, device=dev)
        self.vlm_age = torch.full((n,), vlm_every, dtype=torch.long, device=dev)   # steps since each env's VLM pass
        self.chunk = torch.zeros(n, policy.config["horizon"], ACTION_DIM, device=dev)
        self.chunk_age = torch.full((n,), replan, dtype=torch.long, device=dev)    # actions used of each env's chunk

    def reset(self, env_ids: torch.Tensor):
        """New episodes: their next step runs the VLM and the action head."""
        self.vlm_age[env_ids] = self.vlm_every
        self.chunk_age[env_ids] = self.replan

    def refresh_task(self, head_rgb: torch.Tensor) -> torch.Tensor | None:
        """Run the VLM for the envs whose tokens are vlm_every steps old (or new episodes'); returns their ids."""
        due = (self.vlm_age >= self.vlm_every).nonzero(as_tuple=False).flatten()
        if len(due) == 0:
            return None
        instructions = [self.env.instructions[e] for e in due.tolist()]
        tokens, mask = self.encoders.embed_task(head_rgb[due], instructions)
        words, word_mask = self.encoders.embed_words(instructions)
        self.vlm_tokens[due] = torch.cat([tokens, words], dim=1)
        self.vlm_mask[due] = torch.cat([mask, word_mask], dim=1)
        self.vlm_age[due] = 0
        return due

    @torch.no_grad()
    def act(self, wrist_rgb: torch.Tensor, head_small: torch.Tensor, proprio: torch.Tensor, envs: torch.Tensor) -> torch.Tensor:
        """(N, ACTION_DIM): the next action of each env's chunk; of the envs in the mask `envs`, those that have used
        `replan` actions get a new chunk first. wrist_rgb (N, 2, H, W, 3) and head_small (N, rows, columns, 3;
        FrozenEncoders.head_small) uint8."""
        due = (envs & (self.chunk_age >= self.replan)).nonzero(as_tuple=False).flatten()
        if len(due):
            wrist = wrist_rgb[due].permute(0, 1, 4, 2, 3).float() / 255.0
            head = head_small[due].permute(0, 3, 1, 2).float() / 255.0
            tokens = self.encoders.embed_wrists(wrist)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                chunk = self.policy.act(self.vlm_tokens[due], self.vlm_mask[due], tokens, wrist, head, proprio[due])
            self.chunk[due] = chunk.float()
            self.chunk_age[due] = 0
        rows = torch.arange(self.env.num_envs, device=self.env.device)
        action = self.chunk[rows, self.chunk_age.clamp(max=self.chunk.shape[1] - 1)]
        self.chunk_age[envs] += 1
        return action

    def step(self):
        """After each env step."""
        self.vlm_age += 1
