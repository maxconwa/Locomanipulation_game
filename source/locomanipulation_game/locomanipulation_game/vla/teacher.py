"""The stacking task's privileged teacher: a scripted pick-and-place on the policy's own action interface.

It reads what the policy cannot (the blocks' poses and sizes, which block goes on which, VlaUpperEnv.blocks_in_pelvis)
and answers, for any state, with the 14 actions the policy outputs: per arm a grasp-point position and rotation delta,
then the two gripper commands. So it can drive the env itself or label the states a learning policy visits (DAgger).

Per episode it plans a grasp: the jaws close across one of the block's two horizontal face pairs, the approach leans
`tilts` rad from straight down toward the block as seen from the shoulder (straight down the arms reach half as far);
of the candidates whose grasp and place poses the IK reaches from the ready pose without the arms' spheres colliding,
the one nearest the ready pose in joint space. It plans again when the block has moved while out of the jaws.

Given the plan, the action is a function of the current state alone: where the grasp point is against the block,
how far the jaws have closed, where the block is against its target. No timers and no phase carried from step to
step, so the label in a state a learning policy has wandered into is the one for that state, whatever happened
before, and a policy that sees the state can predict it. (With phases advanced by timers and by gates at the
waypoints, the same state was labelled open in one phase and closed in the next, and a policy hovering 1 cm from a
gate was labelled back and forth across it.) The states, in the order they are tested (`phase` holds each env's):

- the block rests on its target and the jaws are not on it: open (jaws still near closed, hold still), retreat
  (back along the approach), then rest (both arms to the ready pose);
- the jaws are on the block (closed to its width, `closure_on`): closed; lift (straight up until the block's bottom
  clears the target's top), carry and lower, one waypoint that slides from `hover` above the place point down to it
  as the block's horizontal error falls from `over_tol[1]` to `over_tol[0]`; at the place point, open;
- the block is between the open jaws (`grasp_tol`): close, the arm held still;
- else: open; approach and descend, one waypoint that slides from `standoff` back along the approach axis to the
  grasp point as the grasp point's distance from that axis (and its rotation error) falls through `axis_tol`
  (`angle_tol`).

The action is the measured grasp point's error from the waypoint times `gain`, capped (slower where a waypoint is
sliding in), nothing in the hold states. The action's deltas add to the controller's commanded grasp point, which
the measured one follows a step or two behind and sags a few mm from, so this integrates the measured error away:
the command settles where the measured point is on the waypoint. (Steering the commanded point to the waypoint left
the measured one 5-8 mm short, which timers and wide gates then let through; at a gain of 0.5 and above on the
measured error the arm swings across the waypoint, the label's sign flipping with it.) The controller's IK pulls the
arm to the ready posture in the task's null space (DeltaEEArmActionCfg.null_gain), so its joint angles depend on
the grasp point's pose, not on the path there, and the plan's IK, solved with the same pull, finds the same ones.

Plans are made every `plan_every` steps for all the envs waiting, which hold at the ready pose until then: a plan
costs about as much as a policy step whatever the number of envs.
"""

from __future__ import annotations

import math
import torch

from locomanipulation_game.tasks.locomanip_marl.mdp.arm_kinematics import rotation_error

APPROACH, DESCEND, CLOSE, LIFT, CARRY, LOWER, OPEN, RETREAT, REST, WAIT = range(10)
PHASE_NAMES = ("approach", "descend", "close", "lift", "carry", "lower", "open", "retreat", "rest", "wait")

# the jaws' measured closure (MagpieGripperAction.closure) holding a block, by its edge length (m): measured
CLOSURE_SIZES = (0.0325, 0.0375, 0.0425, 0.0475, 0.0525, 0.0575, 0.0625, 0.0675)
CLOSURE_HELD = (0.681, 0.687, 0.665, 0.611, 0.564, 0.539, 0.487, 0.443)


class StackTeacher:
    def __init__(self, env, tilts=(0.5, 0.3, 0.7), gain: float = 0.15, rot_gain: float = 0.15, fast: float = 0.6, slow: float = 0.25, slowest: float = 0.12,
                 standoff: float = 0.08, hover: float = 0.04, tip_reach: float = 0.035,
                 tip_clearance: float = 0.016, place_gap: float = 0.004, shoulder_y: float = 0.2, plan_every: int = 4,
                 axis_tol: tuple[float, float] = (0.008, 0.03), angle_tol: tuple[float, float] = (0.08, 0.25),
                 over_tol: tuple[float, float] = (0.005, 0.012), grasp_tol: tuple[float, float, float] = (0.008, 0.02, 0.008),
                 grasp_angle: float = 0.1, closing_tol: float = 0.015, held_tol: float = 0.03,
                 closure_on: tuple[float, float] = (0.05, 0.10), release_tol: tuple[float, float] = (0.007, 0.004),
                 moved_tol: tuple[float, float] = (0.015, 0.12), release_share: float = 0.25, min_step: float = 0.0):
        self.env, self.tilts = env, tilts
        self.rot_gain = rot_gain
        self.gain = gain  # share of the measured grasp point's distance to the waypoint asked for per step
        self.fast, self.slow, self.slowest = fast, slow, slowest  # the action's largest norm
        self.standoff, self.hover = standoff, hover
        self.tip_reach, self.tip_clearance, self.place_gap = tip_reach, tip_clearance, place_gap
        self.shoulder_y, self.plan_every = shoulder_y, plan_every
        # m from the approach axis (rad of rotation error) where the waypoint is the grasp point / `standoff` back
        self.axis_tol, self.angle_tol = axis_tol, angle_tol
        # m of the held block's horizontal error where the waypoint is the place point / `hover` above it
        self.over_tol = over_tol
        # the block between the open jaws: its offset from where the plan holds it, along the approach, the jaws'
        # closing axis (they centre it) and the third axis (m), and the grasp point's rotation error (rad); once the
        # jaws are closing (closure > 0.1) closing_tol on the first and third; in closed jaws, held_tol overall
        self.grasp_tol, self.grasp_angle, self.closing_tol, self.held_tol = grasp_tol, grasp_angle, closing_tol, held_tol
        self.closure_on = closure_on  # the jaws are on the block: closure within (below, above) of the held one for its size
        # the held block is at the place point: within release_tol[0] m of it horizontally, or release_share of the
        # target's edge where that is more, and no more than release_tol[1] m above it (pressed onto the target counts:
        # with |height| < release_tol[1] a block 1 mm into the target was held there to the episode's end). over_tol
        # widens with the horizontal tolerance.
        self.release_tol, self.release_share = release_tol, release_share
        # m per step: with the block held, the commanded grasp point closes on the waypoint at least this fast (0: off).
        # gain x error alone asks for under 0.5 mm per step in the last 3 mm, less than a learned policy's error on
        # the label. At 2 mm the teacher's success fell from 97.5% to 93.4%, the failures in lower (6 -> 65 of 1500).
        self.min_step = min_step
        self.moved_tol = moved_tol  # a new plan when the block is this far (m, rad of yaw) from where it was planned for
        n, dev = env.num_envs, env.device
        term = env.arm_term
        self.pos_scale, self.rot_scale = term.cfg.pos_scale, term.cfg.rot_scale
        self.rest_q = term._rest.view(n, 2, 7).clone()
        self.rest_p, self.rest_R = term.tcp.poses(self.rest_q)
        self.phase = torch.full((n,), WAIT, dtype=torch.long, device=dev)  # each env's state at the last act()
        self.planned = torch.zeros(n, dtype=torch.bool, device=dev)
        self.grasp_R = self.rest_R[:, 0].clone()          # the grasp point's rotation at the grasp, kept to the release
        self.back = torch.zeros(n, device=dev)            # the grasp point's distance behind the block's centre
        self.plan_p = torch.zeros(n, 3, device=dev)       # the block's position and yaw the plan was made for
        self.plan_yaw = torch.zeros(n, device=dev)
        self.feasible = torch.ones(n, dtype=torch.bool, device=dev)
        self.both_reach = torch.zeros(n, dtype=torch.bool, device=dev)  # either arm reached at the episode's start
        self.reach = torch.zeros(n, 2, dtype=torch.bool, device=dev)    # which arms did
        self._rows = torch.arange(n, device=dev)
        self._steps = 0
        self._sizes = torch.tensor(CLOSURE_SIZES, device=dev)
        self._held = torch.tensor(CLOSURE_HELD, device=dev)
        self.restarts = {"moved": 0}  # plans made again because the block had moved

    def _pair(self):
        env = self.env
        pos, quat = env.blocks_in_pelvis()
        r = self._rows
        return (pos[r, env.pick_slot], quat[r, env.pick_slot], pos[r, env.target_slot],
                env.block_size[r, env.pick_slot], env.block_size[r, env.target_slot])

    def _place_point(self, target_p, size_p, size_t):
        """Where the instructed block's centre rests on the target."""
        place = target_p.clone()
        place[:, 2] += 0.5 * (size_p + size_t) + self.place_gap
        return place

    def _closure_held(self, size: torch.Tensor) -> torch.Tensor:
        """The jaws' closure on a block of this size (CLOSURE_HELD, interpolated)."""
        s = size.clamp(self._sizes[0], self._sizes[-1])
        i = (torch.bucketize(s, self._sizes) - 1).clamp(0, len(self._sizes) - 2)
        w = (s - self._sizes[i]) / (self._sizes[i + 1] - self._sizes[i])
        return self._held[i] + w * (self._held[i + 1] - self._held[i])

    def reset(self, ids: torch.Tensor):
        """The envs ids start over (a new episode): the arm is chosen and the grasp planned.

        The arm: the one that reaches (a plan for it is feasible: the grasp, the place and the point before the
        grasp solved, without collision), either at random where both do, the block's side's where neither does.
        The env's instruction names it (VlaUpperEnv.set_arm)."""
        self.planned[ids] = False
        self.phase[ids] = WAIT
        if len(ids) == 0:
            return
        env = self.env
        plans = [self._plan(ids, torch.full_like(env.arm[ids], a), write=False) for a in (0, 1)]
        reach = torch.stack([plan[0] for plan in plans], dim=1)  # (m, arms)
        either = torch.randint(0, 2, (len(ids),), device=env.device)
        arm = torch.where(reach.all(dim=1), either, torch.where(reach.any(dim=1), reach[:, 1].long(), env.arm[ids]))
        self.both_reach[ids] = reach.all(dim=1)
        self.reach[ids] = reach
        env.set_arm(ids, arm)
        chosen = [torch.where(arm.view(-1, *[1] * (a.dim() - 1)) == 0, a, b) for a, b in zip(*plans)]
        self.feasible[ids], self.grasp_R[ids], self.back[ids], self.plan_p[ids], self.plan_yaw[ids] = chosen
        self.planned[ids] = True

    def _plan(self, ids: torch.Tensor, arm: torch.Tensor | None = None, write: bool = True):
        """Plan the grasp of the envs ids with their arm (or the arms `arm`): whether it is feasible, the grasp
        point's rotation, its distance behind the block's centre, the block's position and yaw; kept if `write`."""
        env, dev = self.env, self.env.device
        term = env.arm_term
        pick_p, pick_q, target_p, size_p, size_t = (v[ids] for v in self._pair())
        arm = env.arm[ids] if arm is None else arm
        m = len(ids)
        yaw = 2.0 * torch.atan2(pick_q[:, 3], pick_q[:, 0])
        side = torch.where(arm == 0, 1.0, -1.0)
        to_block = pick_p[:, :2] - torch.stack([torch.zeros_like(side), side * self.shoulder_y], dim=1)
        down = torch.tensor([0.0, 0.0, -1.0], device=dev)
        Rs = []
        for k in range(4):
            a = yaw + k * math.pi / 2.0
            n_axis = torch.stack([torch.cos(a), torch.sin(a), torch.zeros_like(a)], dim=1)   # the jaws' closing axis
            h = torch.stack([-n_axis[:, 1], n_axis[:, 0], torch.zeros_like(a)], dim=1)
            h = h * torch.sign((h[:, :2] * to_block).sum(dim=1, keepdim=True) + 1e-9)        # away from the shoulder
            for tilt in self.tilts:
                x = math.cos(tilt) * down + math.sin(tilt) * h
                Rs.append(torch.stack([x, n_axis, torch.linalg.cross(x, n_axis)], dim=2))
        R = torch.stack(Rs, dim=1)  # (m, candidates, 3, 3)
        c = R.shape[1]
        # the grasp point: on the approach axis through the block's centre, far enough back that the closed jaws' tips
        # (tip_reach beyond the grasp point; they swing forward as the jaws close) stay tip_clearance above the table
        cos_tilt = -R[..., 2, 0]  # (m, c)
        height = self.tip_reach * cos_tilt + self.tip_clearance              # the grasp point's, above the table
        back = ((height - 0.5 * size_p.unsqueeze(1)) / cos_tilt).clamp(min=0.0)
        grasp = pick_p.unsqueeze(1) - back.unsqueeze(-1) * R[..., 0]         # (m, c, 3)
        place = self._place_point(target_p, size_p, size_t).unsqueeze(1) - back.unsqueeze(-1) * R[..., 0]
        rows = torch.arange(m, device=dev)
        low = term._low[ids].unsqueeze(1).expand(-1, c, -1, -1).reshape(-1, 2, 7)
        high = term._high[ids].unsqueeze(1).expand(-1, c, -1, -1).reshape(-1, 2, 7)
        a_flat = arm.repeat_interleave(c)
        i_flat = torch.arange(m * c, device=dev)

        q0_rest = self.rest_q[ids].unsqueeze(1).expand(-1, c, -1, -1).reshape(-1, 2, 7)

        def solve(points, q0, iters):
            """The planned arm at points (m, candidates, 3) with each candidate's rotation; the other at rest."""
            goal_p = self.rest_p[ids].unsqueeze(1).repeat(1, c, 1, 1)
            goal_R = self.rest_R[ids].unsqueeze(1).repeat(1, c, 1, 1, 1)
            goal_p[rows, :, arm] = points
            goal_R[rows, :, arm] = R
            q, pe, re = term.tcp.solve(goal_p.reshape(-1, 2, 3), goal_R.reshape(-1, 2, 3, 3), q0, low, high, iters=iters,
                                       rot_weight=term.cfg.rot_weight, posture=q0_rest, null_gain=term.cfg.null_gain)
            return q, pe[i_flat, a_flat], re[i_flat, a_flat]

        q0 = q0_rest
        q_grasp, pe_g, re_g = solve(grasp, q0, 50)
        q_place, pe_p, re_p = solve(place, q_grasp, 20)
        q_pre, pe_a, re_a = solve(grasp - self.standoff * R[..., 0], q_grasp, 12)
        ok = (pe_g < 0.005) & (re_g < 0.06) & (pe_p < 0.005) & (re_p < 0.06) & (pe_a < 0.005) & (re_a < 0.06)
        ok &= ~term.tcp.kin.collides(q_grasp) & ~term.tcp.kin.collides(q_place)
        cost = (q_grasp - q0).abs().amax(dim=(1, 2)) + 10.0 * (~ok).float() + 100.0 * (pe_g + pe_p)
        best = cost.view(m, c).argmin(dim=1)
        plan = (ok.view(m, c)[rows, best], R[rows, best], back[rows, best], pick_p, yaw)
        if write:
            self.feasible[ids], self.grasp_R[ids], self.back[ids], self.plan_p[ids], self.plan_yaw[ids] = plan
            self.planned[ids] = True
        return plan

    def act(self) -> torch.Tensor:
        """(N, 14): the action for the current state (`phase` is set to each env's state)."""
        env, dev, n, r = self.env, self.env.device, self.env.num_envs, self._rows
        term = env.arm_term
        tcp_p, tcp_R = term.measured_tcp()
        cmd_p, cmd_R = term.commanded_tcp()
        arm = env.arm
        p, R, cmd = tcp_p[r, arm], tcp_R[r, arm], cmd_p[r, arm]
        closure = env.gripper_term.closure()[r, arm]
        pick_p, pick_q, target_p, size_p, size_t = self._pair()
        place_block = self._place_point(target_p, size_p, size_t)
        off = pick_p - place_block                      # the block from where it rests on the target (+ place_gap)
        off_xy = off[:, :2].norm(dim=-1)

        # the block against the jaws: its offset, in the grasp point's frame, from where the plan holds it
        rel = torch.einsum("nji,nj->ni", R, pick_p - p)
        rel[:, 0] -= self.back
        ang = rotation_error(self.grasp_R, R).norm(dim=-1)
        closing = closure > 0.1
        tol_x = torch.where(closing, self.closing_tol, self.grasp_tol[0])
        tol_z = torch.where(closing, self.closing_tol, self.grasp_tol[2])
        between = (rel[:, 0].abs() < tol_x) & (rel[:, 1].abs() < self.grasp_tol[1]) & (rel[:, 2].abs() < tol_z) & \
            (ang < self.grasp_angle)
        fit = closure - self._closure_held(size_p)
        gripped = (fit > -self.closure_on[0]) & (fit < self.closure_on[1]) & (rel.norm(dim=-1) < self.held_tol)
        stacked = (off_xy < env.cfg.success_xy_share * size_t) & \
            ((off[:, 2] + self.place_gap).abs() < env.cfg.success_z_tol) & ~gripped

        # a new plan where there is none, or the block has moved since while out of the jaws
        yaw = 2.0 * torch.atan2(pick_q[:, 3], pick_q[:, 0])
        turned = torch.remainder(yaw - self.plan_yaw + math.pi / 4.0, math.pi / 2.0) - math.pi / 4.0
        moved = self.planned & ~stacked & ~gripped & ~between & \
            (((pick_p - self.plan_p).norm(dim=-1) > self.moved_tol[0]) | (turned.abs() > self.moved_tol[1]))
        self.restarts["moved"] += int(moved.sum())
        self.planned &= ~moved
        if self._steps % self.plan_every == 0:
            waiting = (~self.planned).nonzero(as_tuple=False).flatten()
            if len(waiting):
                self._plan(waiting)
        self._steps += 1
        waiting = ~self.planned

        approach = self.grasp_R[:, :, 0]
        up = torch.tensor([0.0, 0.0, 1.0], device=dev)
        grasp = pick_p - self.back.unsqueeze(1) * approach      # the grasp point for the block where it is now
        from_grasp = p - grasp
        near = from_grasp.norm(dim=-1) < self.standoff - 0.02

        def ramp(x, tol):
            return ((x - tol[0]) / (tol[1] - tol[0])).clamp(0.0, 1.0)

        # approach and descend: back along the approach axis by how far off it the grasp point is
        along = (from_grasp * approach).sum(dim=-1, keepdim=True)
        off_axis = (from_grasp - along * approach).norm(dim=-1)
        s_in = torch.maximum(ramp(off_axis, self.axis_tol), ramp(ang, self.angle_tol))
        wp_in = grasp - (s_in * self.standoff).unsqueeze(1) * approach
        # lift, carry and lower: the grasp point that puts the held block on the target (by where it sits in the jaws
        # now), raised by how far off the target it is horizontally; until its bottom clears the target's top,
        # straight up
        place = p - off
        release_xy = (self.release_share * size_t).clamp(min=self.release_tol[0])
        wide = release_xy / self.release_tol[0]
        s_over = ((off_xy - self.over_tol[0] * wide) / ((self.over_tol[1] - self.over_tol[0]) * wide)).clamp(0.0, 1.0)
        wp_over = place + (s_over * self.hover).unsqueeze(1) * up
        clear = off[:, 2] + self.place_gap  # the block's bottom above the target's top
        low = (off_xy > 0.02) & (clear < 0.02)
        wp_lift = torch.cat([cmd[:, :2], wp_over[:, 2:3]], dim=1)
        at_place = (off_xy < release_xy) & (off[:, 2] < self.release_tol[1])

        phase = torch.full((n,), APPROACH, dtype=torch.long, device=dev)
        phase[s_in < 1.0] = DESCEND
        phase[between] = CLOSE
        phase[gripped] = CARRY
        phase[gripped & (s_over < 1.0)] = LOWER
        phase[gripped & low] = LIFT
        phase[gripped & at_place] = OPEN
        phase[stacked] = REST
        phase[stacked & near] = RETREAT
        phase[stacked & near & (fit > -2.0 * self.closure_on[0])] = OPEN
        phase[waiting] = WAIT
        self.phase = phase
        self.block_in_jaws, self.angle = rel, ang
        self.block_from_place, self.slide = off, torch.stack([s_in, s_over], dim=1)  # for whoever watches

        rest_p = self.rest_p[r, arm]
        waypoints = torch.stack([
            wp_in,                                    # approach
            wp_in,                                    # descend
            cmd,                                      # close: hold
            wp_lift,                                  # lift
            wp_over,                                  # carry
            wp_over,                                  # lower
            cmd,                                      # open: hold
            grasp - self.standoff * approach,         # retreat
            rest_p,                                   # rest
            rest_p,                                   # wait
        ], dim=1)
        resting = phase >= REST
        goal_p, goal_Rs = self.rest_p.clone(), self.rest_R.clone()
        goal_p[r, arm] = waypoints[r, phase]
        goal_Rs[r, arm] = torch.where(resting.view(n, 1, 1), self.rest_R[r, arm], self.grasp_R)
        err_p, err_R = goal_p - tcp_p, rotation_error(goal_Rs, tcp_R)
        hold = (phase == CLOSE) | (phase == OPEN)
        err_p[r, arm] = torch.where(hold.unsqueeze(1), torch.zeros_like(p), err_p[r, arm])
        err_R[r, arm] = torch.where(hold.unsqueeze(1), torch.zeros_like(p), err_R[r, arm])
        cap = torch.full((n, 2, 1), self.fast, device=dev)
        cap[r, arm, 0] = torch.where(phase == DESCEND, self.slow, torch.where(phase == LOWER, self.slowest, self.fast))

        def toward(err, scale):
            a = err / scale
            return a * (cap / a.norm(dim=-1, keepdim=True).clamp(min=1e-9)).clamp(max=1.0)

        step_p = self.gain * err_p
        # carry and lower: no slower than min_step towards the waypoint, from the commanded point (the measured one
        # trails it, and a floor on the measured error would wind the command up past the waypoint)
        to_wp = waypoints[r, phase] - cmd
        dist = to_wp.norm(dim=-1, keepdim=True)
        floor = to_wp * (dist.clamp(max=self.min_step) / dist.clamp(min=1e-9))
        slow = ((phase == CARRY) | (phase == LOWER)).unsqueeze(1) & \
            (step_p[r, arm].norm(dim=-1, keepdim=True) < dist.clamp(max=self.min_step))
        step_p[r, arm] = torch.where(slow, floor, step_p[r, arm])
        a_p, a_R = toward(step_p, self.pos_scale), toward(self.rot_gain * err_R, self.rot_scale)
        action = torch.cat([a_p, a_R], dim=-1).clamp(-1.0, 1.0)
        grip = -torch.ones(n, 2, device=dev)
        grip[r, arm] = torch.where((phase >= CLOSE) & (phase <= LOWER), 1.0, -1.0)
        return torch.cat([action.reshape(n, 12), grip], dim=1)
