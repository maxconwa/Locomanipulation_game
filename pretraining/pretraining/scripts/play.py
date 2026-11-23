"""Play: a stage-1 run's actors tracking named clips in the Isaac env, the cross-check's rollout, optionally recorded.

  python -m pretraining.scripts.play --run data/runs/amass_s1 --motion-dir data/motions/h12_amass \\
      --clip <name> [<name> ...] [--video] [--out <dir>]

Writes ``<out>/play.csv`` (``crosscheck.csv``'s columns) and, with ``--video``, ``<out>/videos/*.mp4`` from Isaac
Lab's viewer camera (env 0, the first clip) through gymnasium's ``RecordVideo`` (headless, cameras enabled).
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    """The script's own arguments (``main`` adds the app launcher's)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--clip", nargs="+", required=True, help="clip names (file stems)")
    p.add_argument("--seconds", type=float, default=10.0)
    p.add_argument("--video", action="store_true", help="record env 0 (the first clip)")
    p.add_argument("--out", type=Path, default=None, help="default data/pretraining/play/<run name>")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's own arguments (no app launcher's: ``device`` is then cuda:0)."""
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> str:
    """Roll the named clips, write the outputs, and return the summary line. Needs the app (cameras for video)."""
    import pretraining.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.runner import load_actors
    from pretraining.scripts import rollout

    os.environ.setdefault("PRETRAINING_CUDA_GRAPHS", "0")
    out = args.out or REPO / "data" / "pretraining" / "play" / args.run.name
    files = [args.motion_dir / f"{c}.npz" for c in args.clip]
    device = getattr(args, "device", None) or "cuda:0"
    actors = load_actors(args.run, device)
    history_length = next(iter(actors.values())).history_length
    env = rollout.make_env(files, len(files), device, seed=0, render_mode="rgb_array" if args.video else None)
    if args.video:
        import gymnasium as gym

        steps = int(round(args.seconds / env.unwrapped.step_dt))
        env = gym.wrappers.RecordVideo(
            env, video_folder=str(out / "videos"), step_trigger=lambda s: s == 0, video_length=steps,
            disable_logger=True,
        )
    try:
        rows = rollout.rollout(env, actors, history_length, args.seconds)
    finally:
        env.close()
    rollout.write_rows(rows, out / "play.csv")
    for r in rows:
        print(f"[play] {r['clip']}: {r['ended_by']} after {r['time_upright_s']} s, body pos err "
              f"{r['body_pos_err_m']:.3f} m, joint err {r['joint_err_rad']:.3f} rad")
    line = rollout.summary(str(args.run), rows)
    print(f"[play] {line}")
    return line


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
    if args.video:
        args.enable_cameras = True
    app = AppLauncher(args).app
    try:
        run(args)
    except BaseException:
        # A failed run (a CUDA device-side assert in particular) can leave the app unable to shut down: the
        # process then spins with the GPU idle. Report and exit without closing it.
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)
    sys.stdout.flush()  # closing the app ends the process without flushing
    app.close()


if __name__ == "__main__":
    main()
