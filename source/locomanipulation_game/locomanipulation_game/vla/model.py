"""The trained part of the upper-body vision-language-action policy.

    VLM tokens (frozen SmolVLM2: head image + instruction) --linear--\\
    wrist patch tokens (frozen DINOv2, two cameras)        --linear--+
    keypoints (trained KeypointEncoder, three cameras)     --MLP-----+--> memory --> action head --> action chunk
    proprioception (52) + the keypoints' block positions   --MLP-----/  (also added to the head's queries)

The keypoint encoder is the one trained vision part: a small CNN per camera kind whose feature map is queried by
vectors made from the instruction's word embeddings (the frozen VLM's input embeddings), each query's soft-argmax a
keypoint (image position and peak). The frozen encoders alone did not place the instructed block: on 10,000 scenes a
decoder on the VLM's tokens told which arm's side it was on (99%) but placed it no better than the zone's centre
(7 cm), DINOv2's patch tokens with the block's colour and size given likewise (its features carry little colour),
while a CNN on the pixels with the same labels placed it to 0.4 cm. In training the head camera's keypoints also
predict the instructed block's and its target's positions (privileged, an auxiliary loss).

The action head is behind an interface (ActionHead): GaussianChunkHead, a transformer decoder whose `horizon` learned
queries cross-attend the memory and give the mean of each of the chunk's actions, with a learned per-dimension std.
It has a log-probability, so a policy-gradient method (the MAPPO co-training planned with the legs) can fine-tune it.
FlowChunkHead is the same decoder as a flow-matching model: it turns noise into the chunk in a few steps, so where
the teacher's action could go either of two ways it gives one of them, not their mean (the Gaussian head's mean
there was no motion: the policy hovered over the target with the block in its jaws).

An action is 14 numbers in [-1, 1]: per arm a grasp-point position delta and rotation delta (pelvis frame), then the
two gripper commands (mdp/vla.py).
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders import VLM_DIM, VLM_TOKENS, WORD_TOKENS, WRIST_DIM

ACTION_DIM = 14
PROPRIO_DIM = 52


class ActionHead(nn.Module):
    """Memory tokens -> a chunk of actions. Subclasses: loss (training), act (rollouts), log_prob (fine-tuning)."""

    def loss(self, memory, memory_mask, target, valid, direct=None) -> tuple[torch.Tensor, dict[str, float]]:
        raise NotImplementedError

    def act(self, memory, memory_mask, deterministic: bool = True, direct=None) -> torch.Tensor:
        raise NotImplementedError

    def log_prob(self, memory, memory_mask, actions, direct=None) -> torch.Tensor:
        raise NotImplementedError


class GaussianChunkHead(ActionHead):
    def __init__(self, dim: int, horizon: int, layers: int = 4, heads: int = 6, initial_std: float = 0.3):
        super().__init__()
        self.horizon = horizon
        self.queries = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.out = nn.Linear(dim, ACTION_DIM)
        self.log_std = nn.Parameter(torch.full((ACTION_DIM,), math.log(initial_std)))

    def mean(self, memory: torch.Tensor, memory_mask: torch.Tensor, direct: torch.Tensor | None = None) -> torch.Tensor:
        """(B, horizon, ACTION_DIM). The memory's last token is the state's: it is also added to every query.
        `direct` (B, horizon, ACTION_DIM), the policy's state branch, is added to the decoder's output."""
        queries = self.queries.unsqueeze(0) + memory[:, -1:]
        mean = self.out(self.decoder(queries, memory, memory_key_padding_mask=~memory_mask))
        return mean if direct is None else mean + direct

    def loss(self, memory, memory_mask, target, valid, direct=None):
        """target (B, horizon, ACTION_DIM), valid (B, horizon) bool: L1 plus squared error on the mean over the
        valid steps (most of the teacher's deltas are near 0, and L1 alone is content with their median), plus the
        Gaussian likelihood's term for the std alone (the mean detached, so the std fits the residuals without
        reweighting the rest)."""
        mean = self.mean(memory, memory_mask, direct)
        weight = valid.float().unsqueeze(-1)
        count = weight.sum().clamp(min=1.0) * ACTION_DIM
        l1 = ((mean - target).abs() * weight).sum() / count
        l2 = (((mean - target) ** 2) * weight).sum() / count
        var = torch.exp(2.0 * self.log_std)
        nll = ((((mean.detach() - target) ** 2) / (2.0 * var) + self.log_std) * weight).sum() / count
        with torch.no_grad():
            err = (mean - target).abs() * weight
            parts = {"l1": float(l1), "l1_pos": float(err[..., [0, 1, 2, 6, 7, 8]].sum() / (count / ACTION_DIM * 6)),
                     "l1_rot": float(err[..., [3, 4, 5, 9, 10, 11]].sum() / (count / ACTION_DIM * 6)),
                     "l1_grip": float(err[..., 12:].sum() / (count / ACTION_DIM * 2)),
                     "l1_first": float((err[:, 0].sum() / (weight[:, 0].sum().clamp(min=1.0) * ACTION_DIM))),
                     "std": float(self.log_std.exp().mean())}
            # against predicting no motion, and the direction: the first step's position deltas, both arms
            pos = [0, 1, 2, 6, 7, 8]
            parts["zero_pos"] = float((target[..., pos].abs() * weight).sum() / (count / ACTION_DIM * 6))
            t, m = target[:, 0, :12].reshape(-1, 6)[:, :3], mean[:, 0, :12].reshape(-1, 6)[:, :3].float()
            moving = t.norm(dim=-1) > 0.2
            parts["cosine"] = float(F.cosine_similarity(m[moving], t[moving], dim=1).mean()) if moving.any() else 0.0
        return l1 + 2.0 * l2 + 0.05 * nll, parts

    def act(self, memory, memory_mask, deterministic: bool = True, direct=None):
        mean = self.mean(memory, memory_mask, direct)
        if deterministic:
            return mean
        return mean + torch.randn_like(mean) * self.log_std.exp()

    def log_prob(self, memory, memory_mask, actions, direct=None):
        """(B,): the chunk's log-probability."""
        mean = self.mean(memory, memory_mask, direct)
        var = torch.exp(2.0 * self.log_std)
        return (-((actions - mean) ** 2) / (2.0 * var) - self.log_std - 0.5 * math.log(2.0 * math.pi)).sum(dim=(1, 2))


class FlowChunkHead(ActionHead):
    """The chunk by flow matching: the decoder gives the velocity that carries a noise chunk x0 ~ N(0, 1) to the
    teacher's chunk x1 along x_t = (1 - t) x0 + t x1, and a rollout integrates it from noise in `steps` Euler steps.

    Each query is the step's learned vector plus the state token, the noisy chunk's action at that step, the state
    branch's action there (`direct`, detached: what the state alone says, for the flow to keep or leave) and the
    time. The actions are divided by `scale` (the slow position and rotation deltas near a waypoint are about 0.1, the gripper's
    command is +-1), so the noise and the chunk are of one size. The state branch is trained beside the flow by the
    Gaussian head's loss on its own chunk."""

    def __init__(self, dim: int, horizon: int, layers: int = 4, heads: int = 6, steps: int = 10, delta_scale: float = 0.1):
        super().__init__()
        self.horizon, self.steps = horizon, steps
        self.queries = nn.Parameter(torch.randn(horizon, dim) * 0.02)
        self.action_in = nn.Linear(2 * ACTION_DIM, dim)
        self.time_in = nn.Sequential(nn.Linear(64, dim), nn.GELU(), nn.Linear(dim, dim))
        layer = nn.TransformerDecoderLayer(dim, heads, 4 * dim, dropout=0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(layer, layers, norm=nn.LayerNorm(dim))
        self.out = nn.Linear(dim, ACTION_DIM)
        self.register_buffer("scale", torch.tensor([delta_scale] * 12 + [1.0] * 2))
        self.register_buffer("frequencies", torch.exp(torch.linspace(0.0, math.log(1000.0), 32)))

    def velocity(self, memory, memory_mask, x, t, direct=None) -> torch.Tensor:
        """x (B, horizon, ACTION_DIM) the noisy chunk (scaled), t (B,) in [0, 1] -> its velocity (scaled)."""
        d = torch.zeros_like(x) if direct is None else direct.detach().float() / self.scale
        angles = t.float().unsqueeze(1) * self.frequencies
        time = self.time_in(torch.cat([angles.sin(), angles.cos()], dim=1)).unsqueeze(1)
        queries = self.queries.unsqueeze(0) + memory[:, -1:] + self.action_in(torch.cat([x, d], dim=-1)) + time
        return self.out(self.decoder(queries, memory, memory_key_padding_mask=~memory_mask))

    @torch.no_grad()
    def mean(self, memory, memory_mask, direct=None) -> torch.Tensor:
        """A chunk drawn as in a rollout (the head has no mean): (B, horizon, ACTION_DIM)."""
        x = torch.randn(memory.shape[0], self.horizon, ACTION_DIM, device=memory.device)
        for k in range(self.steps):
            t = torch.full((x.shape[0],), k / self.steps, device=x.device)
            x = x + self.velocity(memory, memory_mask, x, t, direct).float() / self.steps
        return x * self.scale

    def loss(self, memory, memory_mask, target, valid, direct=None):
        """target (B, horizon, ACTION_DIM), valid (B, horizon) bool. The parts' errors are of the chunk the
        velocity points to from x_t in one step, x_t + (1 - t) v: over all t, so above a rollout's."""
        x1 = target.float() / self.scale
        x0 = torch.randn_like(x1)
        t = torch.rand(x1.shape[0], device=x1.device)
        tt = t.view(-1, 1, 1)
        xt = (1.0 - tt) * x0 + tt * x1
        v = self.velocity(memory, memory_mask, xt, t, direct).float()
        weight = valid.float().unsqueeze(-1)
        count = weight.sum().clamp(min=1.0) * ACTION_DIM
        flow = (((v - (x1 - x0)) ** 2) * weight).sum() / count
        loss = flow
        if direct is not None:
            loss = loss + (((direct - target).abs() + 2.0 * (direct - target) ** 2) * weight).sum() / count
        with torch.no_grad():
            chunk = (xt + (1.0 - tt) * v) * self.scale
            err = (chunk - target).abs() * weight
            pos = [0, 1, 2, 6, 7, 8]
            parts = {"l1": float(err.sum() / count), "flow": float(flow),
                     "l1_pos": float(err[..., pos].sum() / (count / ACTION_DIM * 6)),
                     "l1_rot": float(err[..., [3, 4, 5, 9, 10, 11]].sum() / (count / ACTION_DIM * 6)),
                     "l1_grip": float(err[..., 12:].sum() / (count / ACTION_DIM * 2)),
                     "l1_first": float((err[:, 0].sum() / (weight[:, 0].sum().clamp(min=1.0) * ACTION_DIM))),
                     "zero_pos": float((target[..., pos].abs() * weight).sum() / (count / ACTION_DIM * 6))}
            a, m = target[:, 0, :12].reshape(-1, 6)[:, :3], chunk[:, 0, :12].reshape(-1, 6)[:, :3]
            moving = a.norm(dim=-1) > 0.2
            parts["cosine"] = float(F.cosine_similarity(m[moving], a[moving], dim=1).mean()) if moving.any() else 0.0
        return loss, parts

    def act(self, memory, memory_mask, deterministic: bool = True, direct=None):
        """A chunk drawn from noise (also when `deterministic`: the flow has no mean to return)."""
        return self.mean(memory, memory_mask, direct)

    def log_prob(self, memory, memory_mask, actions, direct=None):
        raise NotImplementedError("a flow's log-probability needs its ODE's divergence: not written")


HEADS = {"gaussian": GaussianChunkHead, "flow": FlowChunkHead}


def state_features(proprio: torch.Tensor, blocks: torch.Tensor, reach: float = 0.1) -> torch.Tensor:
    """What the action head is conditioned on directly, (B, 82): the proprioception (B, 52: mdp/vla.py vla_proprio,
    then vla_commanded_tcp), the keypoint encoder's estimate of the instructed block's and its target's positions
    (B, 6; m, pelvis frame), and each commanded grasp point's offset to each of the two (24), over `reach`.

    With the estimate only inside the keypoint tokens, the head's deltas had no relation to the teacher's (cosine 0.0
    over 24,000 approach steps at 4,000 training steps) though the estimate itself was within 1 cm: the head did not
    learn the image-to-space geometry a second time. The commanded point's lead over the measured one is not made a
    feature: given it, the head learned to keep moving the way it was already moving."""
    b = proprio.shape[0]
    commanded = proprio[:, 34:52].view(b, 2, 9)[..., :3]                       # (B, arms, 3)
    offsets = blocks.view(b, 1, 2, 3) - commanded.unsqueeze(2)                 # (B, arms, blocks, 3)
    return torch.cat([proprio, blocks / reach, offsets.reshape(b, 12) / reach, offsets.norm(dim=-1).reshape(b, 4) / reach,
                      (offsets[..., :2].norm(dim=-1) < 0.03).float().reshape(b, 4), torch.zeros(b, 4, device=proprio.device)], dim=1)


def _cnn(width: int) -> nn.Sequential:
    """Four stride-2 convolutions: an image to a width-channel map at 1/16 of its size."""
    channels = [3, 32, 64, 96, width]
    return nn.Sequential(*[m for i in range(4) for m in (
        nn.Conv2d(channels[i], channels[i + 1], 5, 2, 2), nn.GroupNorm(8, channels[i + 1]), nn.GELU())])


class KeypointEncoder(nn.Module):
    """Keypoints of the head image and the two wrist images, queried by the instruction's words (module doc).

    Per camera, `keypoints` query vectors (an MLP of the word embeddings) are each matched against the camera's CNN
    feature map; a keypoint is the match's soft-argmax position (x, y in [-1, 1]) and its peak probability."""

    def __init__(self, keypoints: int = 8, width: int = 128, cameras: int = 3):
        super().__init__()
        self.keypoints, self.width, self.cameras = keypoints, width, cameras
        self.head_cnn, self.wrist_cnn = _cnn(width), _cnn(width)
        self.word_in = nn.Sequential(nn.LayerNorm(VLM_DIM), nn.Linear(VLM_DIM, 64))
        self.queries = nn.Sequential(nn.Linear(WORD_TOKENS * 64, 512), nn.GELU(), nn.Linear(512, cameras * keypoints * width))
        # the instructed block's and its target's positions (pelvis frame) from the head camera's keypoints
        self.blocks = nn.Sequential(nn.Linear(keypoints * 3, 256), nn.GELU(), nn.Linear(256, 6))

    @staticmethod
    def _grid(h: int, w: int, device) -> torch.Tensor:
        y, x = torch.meshgrid(torch.linspace(-1, 1, h, device=device), torch.linspace(-1, 1, w, device=device), indexing="ij")
        return torch.stack([x, y], dim=-1).reshape(-1, 2)

    def forward(self, head: torch.Tensor, wrists: torch.Tensor, words: torch.Tensor, word_mask: torch.Tensor) -> torch.Tensor:
        """head (B, 3, H, W) and wrists (B, 2, 3, h, w) in [0, 1], words (B, WORD_TOKENS, VLM_DIM) and their mask
        -> keypoints (B, cameras, keypoints, 3): head, left wrist, right wrist."""
        b = head.shape[0]
        w = self.word_in(words.float()) * word_mask.unsqueeze(-1)
        q = self.queries(w.flatten(1)).view(b, self.cameras, self.keypoints, self.width)
        maps = [self.head_cnn(head)] + list(self.wrist_cnn(wrists.flatten(0, 1)).unflatten(0, (b, 2)).unbind(1))
        out = []
        for cam, f in enumerate(maps):
            match = torch.einsum("bkc,bchw->bkhw", q[:, cam].float(), f.float()) / self.width ** 0.5
            prob = match.flatten(2).softmax(dim=-1)
            xy = prob @ self._grid(f.shape[2], f.shape[3], f.device)
            out.append(torch.cat([xy, prob.amax(dim=-1, keepdim=True)], dim=-1))
        return torch.stack(out, dim=1)


class VlaPolicy(nn.Module):
    def __init__(self, wrist_tokens: int, dim: int = 384, horizon: int = 16, head: str = "gaussian", cameras: int = 2,
                 vlm_tokens: int = VLM_TOKENS, keypoints: int = 8):
        super().__init__()
        self.config = {"wrist_tokens": wrist_tokens, "dim": dim, "horizon": horizon, "head": head, "cameras": cameras,
                       "vlm_tokens": vlm_tokens, "keypoints": keypoints}
        self.vlm_tokens = vlm_tokens
        self.vlm_in = nn.Sequential(nn.LayerNorm(VLM_DIM), nn.Linear(VLM_DIM, dim))
        self.vlm_pos = nn.Parameter(torch.randn(vlm_tokens, dim) * 0.02)
        self.wrist_in = nn.Sequential(nn.LayerNorm(WRIST_DIM), nn.Linear(WRIST_DIM, dim))
        self.wrist_pos = nn.Parameter(torch.randn(cameras, wrist_tokens, dim) * 0.02)
        self.proprio_in = nn.Sequential(nn.Linear(PROPRIO_DIM + 30, dim), nn.GELU(), nn.Linear(dim, dim), nn.GELU(),
                                        nn.Linear(dim, dim))
        self.keypoint_encoder = KeypointEncoder(keypoints)
        self.keypoint_in = nn.Sequential(nn.Linear(3, dim), nn.GELU(), nn.Linear(dim, dim))
        self.keypoint_pos = nn.Parameter(torch.randn(3 * keypoints, dim) * 0.02)
        # the state branch: the chunk from the state features alone, added to the action head's output. A plain MLP
        # on standardized inputs fits the teacher from these features in a few thousand steps (offline: its
        # deltas' cosine with the teacher's 0.90); the transformer head given the same features as a token did not
        # in 5,000, so it is left the part the images add.
        self.state_norm = nn.BatchNorm1d(PROPRIO_DIM + 30, affine=False)
        self.state_branch = nn.Sequential(nn.Linear(PROPRIO_DIM + 30, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(),
                                          nn.Linear(512, 512), nn.GELU(), nn.Linear(512, horizon * ACTION_DIM))
        self.head: ActionHead = HEADS[head](dim, horizon)

    def memory(self, task_tokens, task_mask, wrist_tokens, wrist_rgb, head_rgb, proprio):
        """task_tokens (B, vlm_tokens + WORD_TOKENS, VLM_DIM): the VLM's tokens, then the instruction's word
        embeddings, and their mask; wrist_tokens (B, cameras, Lw, WRIST_DIM); wrist_rgb (B, 2, 3, h, w) and head_rgb
        (B, 3, H, W) in [0, 1]; proprio (B, PROPRIO_DIM) -> the memory (B, L, dim), its mask, the blocks' estimated
        positions (B, 6), the state branch's chunk."""
        b = proprio.shape[0]
        n = self.vlm_tokens
        vlm = self.vlm_in(task_tokens[:, :n].float()) + self.vlm_pos
        wrist = (self.wrist_in(wrist_tokens.float()) + self.wrist_pos).flatten(1, 2)
        keypoints = self.keypoint_encoder(head_rgb, wrist_rgb, task_tokens[:, n:], task_mask[:, n:])
        points = self.keypoint_in(keypoints.flatten(1, 2).float()) + self.keypoint_pos
        # the head camera's keypoints' estimate of the two blocks' positions: the auxiliary loss's output, and
        # (without its gradient) the action head's input
        blocks = self.keypoint_encoder.blocks(keypoints[:, 0].flatten(1).float())
        state = self.state_norm(state_features(proprio, blocks.detach()).float())
        direct = self.state_branch(state).view(b, -1, ACTION_DIM)
        prop = self.proprio_in(state).unsqueeze(1)
        ones = torch.ones(b, wrist.shape[1] + points.shape[1] + 1, dtype=torch.bool, device=proprio.device)
        return torch.cat([vlm, wrist, points, prop], dim=1), torch.cat([task_mask[:, :n], ones], dim=1), blocks, direct

    def loss(self, task_tokens, task_mask, wrist_tokens, wrist_rgb, head_rgb, proprio, target, valid, blocks,
             blocks_weight: float = 10.0):
        """The action head's loss plus the auxiliary one: the head camera's keypoints' prediction of blocks (B, 6),
        the instructed block's and its target's positions (m, pelvis frame), L1 x blocks_weight."""
        memory, mask, estimate, direct = self.memory(task_tokens, task_mask, wrist_tokens, wrist_rgb, head_rgb, proprio)
        loss, parts = self.head.loss(memory, mask, target, valid, direct)
        error = (estimate - blocks).abs()
        parts["blocks_cm"] = float(error.view(-1, 2, 3)[..., :2].norm(dim=-1).mean()) * 100.0
        return loss + blocks_weight * error.mean(), parts

    @torch.no_grad()
    def act(self, task_tokens, task_mask, wrist_tokens, wrist_rgb, head_rgb, proprio, deterministic: bool = True):
        """(B, horizon, ACTION_DIM), clamped to the actions' range."""
        memory, mask, _, direct = self.memory(task_tokens, task_mask, wrist_tokens, wrist_rgb, head_rgb, proprio)
        return self.head.act(memory, mask, deterministic, direct).clamp(-1.0, 1.0)


def augment_images(images: torch.Tensor, shift: int = 3, jitter: float = 0.15) -> torch.Tensor:
    """Training augmentation of images (B, cameras, 3, H, W) in [0, 1]: a random shift of up to `shift` px per
    image (edge-padded) and a random brightness and contrast change of up to `jitter`."""
    b, c, _, h, w = images.shape
    x = F.pad(images.flatten(0, 1), (shift, shift, shift, shift), mode="replicate")
    n = x.shape[0]
    dx = torch.randint(0, 2 * shift + 1, (n,), device=x.device)
    dy = torch.randint(0, 2 * shift + 1, (n,), device=x.device)
    cols = (torch.arange(w, device=x.device).view(1, 1, w) + dx.view(n, 1, 1)).expand(n, h, w)
    rows = (torch.arange(h, device=x.device).view(1, h, 1) + dy.view(n, 1, 1)).expand(n, h, w)
    x = x[torch.arange(n, device=x.device).view(n, 1, 1), :, rows, cols].permute(0, 3, 1, 2)
    gain = 1.0 + (torch.rand(n, 1, 1, 1, device=x.device) * 2.0 - 1.0) * jitter
    bias = (torch.rand(n, 1, 1, 1, device=x.device) * 2.0 - 1.0) * jitter
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    return ((x - mean) * gain + mean + bias).clamp(0.0, 1.0).view(b, c, 3, h, w)
