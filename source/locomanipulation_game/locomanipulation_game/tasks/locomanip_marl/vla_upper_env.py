"""LocoManip-VlaUpper-v0's env: LocoManipMarlEnv with the stacking task on top (vla_upper_env_cfg.py).

Each reset draws an episode's configuration (how many blocks, the instructed block's and its target's size names,
which arm's zone they are in) by success-guided sampling (SuccessGuidedSampler), puts the blocks on the table and
words the instruction. Each step checks whether the instructed block rests on its target, released; holding that for
cfg.success_hold_s ends the episode as a success. The policy's images and instruction are read from the env
(camera_images, instructions); a privileged teacher reads the blocks (blocks_in_pelvis).
"""

from __future__ import annotations

import math
import torch
from collections.abc import Sequence

from isaaclab.utils.math import quat_from_euler_xyz, subtract_frame_transforms, yaw_quat

from .locomanip_marl_env import LocoManipMarlEnv
from .mdp import vla
from .vla_upper_env_cfg import ARMS_ACTION, BLOCK_NAMES, CAMERA_NAMES, GRIPPER_ACTION, TABLE_SIZE, TABLE_TOP, VlaUpperEnvCfg


class SuccessGuidedSampler:
    """Draws episode configurations the policy succeeds at about `target` of the time.

    Each configuration bin keeps an exponential mean of its episodes' successes (from 0.5). A bin is drawn with
    the weight of a Beta density at its rate, rate^(a - 1) x (1 - rate)^(b - 1), whose mode is `target` and whose
    a + b is `concentration` (a = target x (concentration - 2) + 1): most at the target success rate, none where
    the policy always succeeds or always fails; a share `floor` of the draws is uniform over the bins, so every bin
    keeps being tried and its rate stays current. (The weight was rate x (1 - rate), the Beta of mode 0.5 and
    concentration 4: the right arm's bins, a little behind the left's at 8,000 steps, got 28% of the episodes
    while the left's went to 74% success.)
    """

    def __init__(self, shape: tuple[int, ...], floor: float, decay: float, device: str, target: float = 0.4,
                 concentration: float = 4.0):
        self.shape, self.floor, self.decay = shape, floor, decay
        self.alpha = target * (concentration - 2.0) + 1.0
        self.beta = (1.0 - target) * (concentration - 2.0) + 1.0
        self.rate = torch.full(shape, 0.5, device=device)
        self.count = torch.zeros(shape, dtype=torch.long, device=device)

    def weights(self) -> torch.Tensor:
        """The bins' draw probabilities, flat."""
        w = (self.rate ** (self.alpha - 1.0) * (1.0 - self.rate) ** (self.beta - 1.0)).flatten()
        n = w.numel()
        guided = w / w.sum() if float(w.sum()) > 1e-9 else torch.full_like(w, 1.0 / n)
        return (1.0 - self.floor) * guided + self.floor / n

    def sample(self, n: int) -> torch.Tensor:
        """(n, len(shape)): bin indices."""
        flat = torch.multinomial(self.weights(), n, replacement=True)
        return torch.stack(torch.unravel_index(flat, self.shape), dim=1)

    def update(self, bins: torch.Tensor, success: torch.Tensor):
        """bins (n, len(shape)), success (n,) bool: the ended episodes."""
        for b, s in zip(bins.tolist(), success.float().tolist()):
            b = tuple(b)
            self.rate[b] = self.decay * self.rate[b] + (1.0 - self.decay) * s
            self.count[b] += 1

    def state_dict(self) -> dict:
        return {"rate": self.rate.clone(), "count": self.count.clone()}

    def load_state_dict(self, state: dict):
        self.rate[:] = state["rate"].to(self.rate.device)
        self.count[:] = state["count"].to(self.count.device)


class VlaUpperEnv(LocoManipMarlEnv):
    cfg: VlaUpperEnvCfg

    def __init__(self, cfg: VlaUpperEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        n, dev = self.num_envs, self.device
        self.blocks = [self.scene[name] for name in BLOCK_NAMES]
        self.table = self.scene["table"]
        self.arm_term = self.action_manager.get_term(ARMS_ACTION)
        self.gripper_term = self.action_manager.get_term(GRIPPER_ACTION)
        sizes, colors = vla.block_layout(n, cfg.block_seed)
        self.block_size, self.block_color = sizes.to(dev), colors.to(dev)  # (N, slots)
        self.block_class = torch.tensor(vla.BLOCK_SLOT_CLASSES, device=dev).expand(n, -1)
        self._color_names = list(vla.BLOCK_COLORS)
        self._block_shaders = vla.block_shader_inputs(self, BLOCK_NAMES) if cfg.block_colors_per_episode else None
        self.visuals = vla.VisualRandomizer(self, cfg.visuals, BLOCK_NAMES) if cfg.visuals is not None else None
        self._parked = torch.stack([b.data.default_root_state[:, :7].clone() for b in self.blocks], dim=1)
        # the episode: the blocks on the table, the instructed block, its target, the arm whose zone they are in
        self.active = torch.zeros(n, len(self.blocks), dtype=torch.bool, device=dev)
        self.pick_slot = torch.zeros(n, dtype=torch.long, device=dev)
        self.target_slot = torch.ones(n, dtype=torch.long, device=dev)
        self.arm = torch.zeros(n, dtype=torch.long, device=dev)
        self.table_top = torch.full((n,), TABLE_TOP, device=dev)  # above the env's origin
        self.instructions = [""] * n
        # bins: blocks on the table, the instructed block's size name, its target's, the pair's side (0 left: that arm's
        # episode, unless the teacher finds that the other arm reaches too and picks it)
        low, high = cfg.num_blocks_range
        names = len(vla.BLOCK_SIZE_NAMES)
        self.sampler = SuccessGuidedSampler((high - low + 1, names, names, 2), cfg.sampler_floor, cfg.sampler_decay, dev,
                                            cfg.sampler_target, cfg.sampler_concentration)
        self.bins = torch.zeros(n, 4, dtype=torch.long, device=dev)
        self.sampler_mask = torch.ones(n, dtype=torch.bool, device=dev)
        """Envs whose episodes' outcomes update the sampler (the trainer clears the teacher-driven ones)."""
        self.end_now = torch.zeros(n, dtype=torch.bool, device=dev)
        """Envs whose episode ends at this step whatever its state (the trainer's scene envs); cleared by the step."""
        self.stack_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.block_off_table = torch.zeros(n, dtype=torch.bool, device=dev)
        self._held_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self._hold_steps = max(1, round(cfg.success_hold_s / self.step_dt))
        # the last step's ended episodes, for whoever drives the env: which, whether stacked, their bins
        self.ended = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ended_success = torch.zeros(n, dtype=torch.bool, device=dev)
        self.ended_bins = self.bins.clone()

    """
    What the policy and the teacher read.
    """

    def camera_images(self) -> dict[str, torch.Tensor]:
        """Camera name -> (N, H, W, 3) uint8 RGB."""
        return {name: self.scene[name].data.output["rgb"][..., :3] for name in CAMERA_NAMES}

    def blocks_in_pelvis(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Every block's position (N, slots, 3) and quaternion (N, slots, 4) in the pelvis frame (privileged)."""
        robot = self._robot
        pos = torch.stack([b.data.root_pos_w for b in self.blocks], dim=1)
        quat = torch.stack([b.data.root_quat_w for b in self.blocks], dim=1)
        slots = pos.shape[1]
        root_pos = robot.data.root_pos_w.unsqueeze(1).expand(-1, slots, -1)
        root_quat = robot.data.root_quat_w.unsqueeze(1).expand(-1, slots, -1)
        p, q = subtract_frame_transforms(root_pos.reshape(-1, 3), root_quat.reshape(-1, 4), pos.reshape(-1, 3), quat.reshape(-1, 4))
        return p.view(-1, slots, 3), q.view(-1, slots, 4)

    def _slot(self, values: torch.Tensor, slot: torch.Tensor) -> torch.Tensor:
        return values[torch.arange(self.num_envs, device=self.device), slot]

    """
    DirectMARLEnv hooks.
    """

    def _get_dones(self):
        self._update_success()
        terminated, time_outs = super()._get_dones()
        if self.end_now.any():
            time_outs = {name: value | self.end_now for name, value in time_outs.items()}
            self.end_now[:] = False
        agent = self.cfg.possible_agents[0]
        self.ended = terminated[agent] | time_outs[agent]
        self.ended_success = self.stack_success & self.ended
        self.ended_bins = self.bins.clone()
        counted = self.ended & self.sampler_mask
        if counted.any():
            self.sampler.update(self.bins[counted], self.stack_success[counted])
        return terminated, time_outs

    def _update_success(self):
        cfg = self.cfg
        pos, _ = self.blocks_in_pelvis()
        pick, target = self._slot(pos, self.pick_slot), self._slot(pos, self.target_slot)
        size_p, size_t = self._slot(self.block_size, self.pick_slot), self._slot(self.block_size, self.target_slot)
        over = (pick[:, :2] - target[:, :2]).norm(dim=-1) < cfg.success_xy_share * size_t
        resting = (pick[:, 2] - target[:, 2] - 0.5 * (size_p + size_t)).abs() < cfg.success_z_tol
        speed = torch.stack([b.data.root_lin_vel_w for b in self.blocks], dim=1)
        still = self._slot(speed, self.pick_slot).norm(dim=-1) < cfg.success_max_speed
        tcp, _ = self.arm_term.measured_tcp()
        released = ((tcp - pick.unsqueeze(1)).norm(dim=-1) > cfg.success_release_distance).all(dim=1)
        stacked = over & resting & still & released
        low = torch.minimum(pick[:, 2], target[:, 2]) + self._robot.data.root_pos_w[:, 2] - self.scene.env_origins[:, 2]
        self.block_off_table = low < self.table_top - 0.02
        self._held_steps = torch.where(stacked, self._held_steps + 1, torch.zeros_like(self._held_steps))
        self.stack_success = self._held_steps >= self._hold_steps

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device).long()
        self._held_steps[ids] = 0
        self.stack_success[ids] = False
        self._draw_episodes(ids)
        # render the reset envs' new scene: DirectMARLEnv.step doesn't, and their first images would be the last
        # episode's (the reset marked their cameras outdated, so the next read fetches this frame)
        self.scene.write_data_to_sim()
        self.sim.forward()
        self.sim.render()

    """
    The episode's configuration.
    """

    def _draw_episodes(self, ids: torch.Tensor):
        cfg, dev, n = self.cfg, self.device, len(ids)
        slots = len(self.blocks)
        bins = self.sampler.sample(n)
        self.bins[ids] = bins
        count = bins[:, 0] + cfg.num_blocks_range[0]
        arm = bins[:, 3]
        # the instructed block: either slot of its size name; its target: a slot of the target's name, not the same
        classes = self.block_class[ids]
        noise = torch.rand(n, slots, device=dev)
        pick = torch.where(classes == bins[:, 1:2], noise, -1.0).argmax(dim=1)
        noise = torch.rand(n, slots, device=dev)
        noise[torch.arange(n, device=dev), pick] = -2.0
        target = torch.where(classes == bins[:, 2:3], noise, -1.0).argmax(dim=1)
        # the others on the table: the first count - 2 of a random order of the remaining slots
        order = torch.rand(n, slots, device=dev)
        rows = torch.arange(n, device=dev)
        order[rows, pick] = -2.0
        order[rows, target] = -1.0
        rank = order.argsort(dim=1).argsort(dim=1)  # the pair first, then the rest in random order
        active = rank < count.unsqueeze(1)
        # positions in the pelvis's yaw frame: the pair in the arm's zone (or the shared one), the others in either;
        # drawn again while
        # any two on the table are closer than block_min_distance
        side_pair = torch.where(arm == 0, 1.0, -1.0)
        xy = torch.zeros(n, slots, 2, device=dev)
        shared = torch.rand(n, device=dev) < cfg.shared_zone_share
        redo = torch.ones(n, dtype=torch.bool, device=dev)
        for _ in range(200):
            m = int(redo.sum())
            if m == 0:
                break
            side = torch.where(torch.rand(m, slots, device=dev) < 0.5, 1.0, -1.0)
            r = torch.arange(m, device=dev)
            side[r, pick[redo]] = side_pair[redo]
            side[r, target[redo]] = side_pair[redo]
            x = torch.empty(m, slots, device=dev).uniform_(*cfg.spare_zone_x)
            y = torch.empty(m, slots, device=dev).uniform_(*cfg.spare_zone_y) * side
            for slot in (pick[redo], target[redo]):
                x[r, slot] = torch.empty(m, device=dev).uniform_(*cfg.zone_x)
                y[r, slot] = torch.empty(m, device=dev).uniform_(*cfg.zone_y) * side_pair[redo]
                x[r, slot] = torch.where(shared[redo], torch.empty(m, device=dev).uniform_(*cfg.shared_zone_x), x[r, slot])
                y[r, slot] = torch.where(shared[redo], torch.empty(m, device=dev).uniform_(*cfg.shared_zone_y), y[r, slot])
            xy[redo] = torch.stack([x, y], dim=-1)
            dist = torch.cdist(xy, xy) + 10.0 * torch.eye(slots, device=dev)
            dist = dist + 10.0 * (~(active.unsqueeze(1) & active.unsqueeze(2))).float()
            redo = dist.amin(dim=(1, 2)) < cfg.block_min_distance
        self.active[ids], self.pick_slot[ids], self.target_slot[ids], self.arm[ids] = active, pick, target, arm

        # the table's height, then the blocks: on the table under the robot's yaw frame, or parked
        robot = self._robot
        origins = self.scene.env_origins[ids]
        top = TABLE_TOP + torch.empty(n, device=dev).uniform_(*cfg.table_height_range)
        self.table_top[ids] = top
        table_pose = self.table.data.default_root_state[ids, :7].clone()
        table_pose[:, :3] += origins
        table_pose[:, 2] = origins[:, 2] + top - 0.5 * TABLE_SIZE[2]
        self.table.write_root_pose_to_sim(table_pose, env_ids=ids)
        root_pos = robot.data.root_pos_w[ids]
        yaw = yaw_quat(robot.data.root_quat_w[ids])
        heading = 2.0 * torch.atan2(yaw[:, 3], yaw[:, 0])
        cos, sin = torch.cos(heading).unsqueeze(1), torch.sin(heading).unsqueeze(1)
        world_x = root_pos[:, 0:1] + cos * xy[..., 0] - sin * xy[..., 1]
        world_y = root_pos[:, 1:2] + sin * xy[..., 0] + cos * xy[..., 1]
        sizes = self.block_size[ids]
        world_z = (origins[:, 2] + top).unsqueeze(1) + 0.5 * sizes + 0.002
        spin = torch.rand(n, slots, device=dev) * 2.0 * math.pi
        zeros = torch.zeros_like(spin)
        quat = quat_from_euler_xyz(zeros.flatten(), zeros.flatten(), spin.flatten()).view(n, slots, 4)
        placed = torch.cat([torch.stack([world_x, world_y, world_z], dim=-1), quat], dim=-1)
        parked = self._parked[ids].clone()
        parked[..., :3] += origins.unsqueeze(1)
        parked[..., 2] = origins[:, 2:3] + 0.5 * sizes + 0.002
        pose = torch.where(active.unsqueeze(-1), placed, parked)
        still = torch.zeros(n, 6, device=dev)
        for slot, block in enumerate(self.blocks):
            block.write_root_pose_to_sim(pose[:, slot], env_ids=ids)
            block.write_root_velocity_to_sim(still, env_ids=ids)

        if self._block_shaders is not None:
            # new colours: two slots of the same size name never share one (block_layout's rule)
            palette = len(self._color_names)
            for c in range(len(names := vla.BLOCK_SIZE_NAMES)):
                pair = (self.block_class[0] == c).nonzero().flatten()
                order = torch.rand(n, palette, device=dev).argsort(dim=1)[:, :len(pair)]
                self.block_color[ids.unsqueeze(1), pair.unsqueeze(0)] = order
            vla.color_blocks(self._block_shaders, ids.tolist(), self.block_color)
        if self.visuals is not None:
            self.visuals.randomize(ids.tolist(), self.block_color)
        self._write_instructions(ids)

    def set_arm(self, ids: torch.Tensor, arm: torch.Tensor):
        """The arm (0 left, 1 right) the envs ids' episodes are to be done with: whoever knows which arm reaches
        the blocks says so after the reset (the teacher; until then it is the blocks' side's)."""
        self.arm[ids] = arm
        self._write_instructions(ids)

    def _write_instructions(self, ids: torch.Tensor):
        colors, names = self.block_color, vla.BLOCK_SIZE_NAMES
        for e in ids.tolist():
            p, t = int(self.pick_slot[e]), int(self.target_slot[e])
            self.instructions[e] = (
                f"stack the {names[int(self.block_class[e, p])]} {self._color_names[int(colors[e, p])]} block on the"
                f" {names[int(self.block_class[e, t])]} {self._color_names[int(colors[e, t])]} block"
                f" with the {('left', 'right')[int(self.arm[e])]} arm")
