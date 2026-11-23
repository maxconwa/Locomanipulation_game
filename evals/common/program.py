"""THE command program: what the videos (Isaac, MuJoCo) and the GOLEM deploy run send, so a policy is seen and
scored doing the same things everywhere.

Legs active:
  hang     HANG_S on the hoist, the feet HANG_LIFT off the ground, the policy running (zero command)
  lower    the hoist lowers the robot onto its feet over LOWER_S (the rope still carrying the weight)
  settle   SETTLE_S on its feet under the rope
  release  the rope's share ramps to 0 over RELEASE_S; then STAND_S standing free
  moves    forward, backward, side step left, side step right, forward-left, back-right, turn in place left and right,
           walk and turn; each MOVE_S, each followed by a STOP_S stop (zero command)
  Speeds are shares of the run's command range (a full stick): MOVES' (vx, vy, wz) x the range edge per axis.
Arms active: a goal every GOAL_EVERY_S from the arms' own goal table (the backend draws it), the rest pose at the end.
Arms only (pelvis pinned): no hoist, no moves: ARMS_ONLY_GOALS goals of GOAL_EVERY_S.
"""

from __future__ import annotations

from dataclasses import dataclass

HANG_S, LOWER_S, SETTLE_S, RELEASE_S, STAND_S = 3.0, 3.0, 1.5, 1.0, 3.0
HANG_LIFT, HANG_SLACK = 0.06, 0.10   # m: the feet off the ground while hung; the hoist lowers lift + slack
MOVE_S, STOP_S = 4.0, 2.0
GOAL_EVERY_S = 5.0
ARMS_ONLY_GOALS = 8

# (name, vx, vy, wz) as shares of the run's range edges
MOVES = [
    ("forward", 0.7, 0.0, 0.0),
    ("backward", -0.5, 0.0, 0.0),
    ("side step left", 0.0, 0.6, 0.0),
    ("side step right", 0.0, -0.6, 0.0),
    ("angled forward-left", 0.5, 0.45, 0.0),
    ("angled back-right", -0.4, -0.45, 0.0),
    ("turn in place left", 0.0, 0.0, 0.6),
    ("turn in place right", 0.0, 0.0, -0.6),
    ("walk and turn", 0.6, 0.0, 0.5),
]


@dataclass
class Segment:
    name: str
    kind: str          # hang | lower | settle | release | stand | move | stop | goals
    start: float       # s from the program's start
    seconds: float
    cmd: tuple[float, float, float] = (0.0, 0.0, 0.0)


def segments(legs: bool, full: tuple[float, float, float]) -> list[Segment]:
    """The program's segments. full: the run's range edge per axis (vx, vy, wz)."""
    out, t = [], 0.0

    def add(name, kind, seconds, cmd=(0.0, 0.0, 0.0)):
        nonlocal t
        out.append(Segment(name, kind, t, seconds, tuple(float(c) for c in cmd)))
        t += seconds
    if not legs:
        add("arm goals", "goals", ARMS_ONLY_GOALS * GOAL_EVERY_S)
        return out
    add("hung on the hoist, feet off the ground", "hang", HANG_S)
    add("lowering onto its feet", "lower", LOWER_S)
    add("on its feet under the rope", "settle", SETTLE_S)
    add("releasing the rope", "release", RELEASE_S)
    add("standing free", "stand", STAND_S)
    for name, vx, vy, wz in MOVES:
        add(name, "move", MOVE_S, (vx * full[0], vy * full[1], wz * full[2]))
        add("stop", "stop", STOP_S)
    return out


def duration(segs: list[Segment]) -> float:
    return segs[-1].start + segs[-1].seconds


def at(segs: list[Segment], t: float) -> Segment:
    for s in segs:
        if s.start <= t < s.start + s.seconds:
            return s
    return segs[-1]


def rope_share(t: float) -> float:
    """The weight's share the hoist carries at time t (legs program): all of it until the release, then a ramp to 0."""
    rel = HANG_S + LOWER_S + SETTLE_S
    return 1.0 - min(max((t - rel) / RELEASE_S, 0.0), 1.0)


def anchor_drop(t: float) -> float:
    """How far (m) the hoist has lowered the hook at time t."""
    return (HANG_LIFT + HANG_SLACK) * min(max((t - HANG_S) / LOWER_S, 0.0), 1.0)


def goal_times(segs: list[Segment], legs: bool) -> list[float]:
    """When a new arm goal starts (arms active): every GOAL_EVERY_S from the robot standing free (or from 0 pinned)."""
    start = 0.0 if not legs else next(s.start for s in segs if s.kind == "stand")
    end = duration(segs) - (STOP_S if legs else 0.0)
    out, t = [], start
    while t + GOAL_EVERY_S <= end + 1e-9:
        out.append(t)
        t += GOAL_EVERY_S
    return out
