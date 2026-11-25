"""Cross-check: how well a stage-1 run's actors (h12_train's or this package's) track clips in the Isaac env.

  python -m pretraining.scripts.crosscheck --run data/runs/amass_s1 --motion-dir data/motions/h12_amass \\
      --clips 40 --out data/pretraining/crosscheck/amass_s1

Rolls each chosen clip from its first frame for its length (at most ``--seconds``) with no perturbation and no
noise (``rollout.py``) and writes ``<out>/crosscheck.csv`` (one row per clip), ``<out>/clips.txt`` (the clip names,
for ``--clips-file``) and ``<out>/summary.txt``. Clips: ``--clips-file``'s, else the ``--include`` names then clips
drawn with ``--seed``.
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
    p.add_argument("--run", type=Path, required=True, help="a stage-1 run directory (actors.pt, run.json)")
    p.add_argument("--motion-dir", type=Path, required=True)
    p.add_argument("--clips", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--include", nargs="*", default=[], help="clip names (file stems) to roll first")
    p.add_argument("--clips-file", type=Path, default=None, help="roll the clips named in this file instead")
    p.add_argument("--num-envs", type=int, default=None, help="envs rolled at once (default: one per clip)")
    p.add_argument("--seconds", type=float, default=10.0, help="score at most this much of each clip")
    p.add_argument("--out", type=Path, default=None, help="default data/pretraining/crosscheck/<run name>")
    return p


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the script's own arguments (no app launcher's: ``device`` is then cuda:0)."""
    return build_parser().parse_args(argv)


def run(args: argparse.Namespace) -> str:
    """Roll the clips with the run's actors, write the outputs, and return the summary line. Needs the app."""
    import pretraining.tasks  # noqa: F401  (registers the gym ids)
    from pretraining.marl.runner import load_actors
    from pretraining.scripts import rollout

    os.environ.setdefault("PRETRAINING_CUDA_GRAPHS", "0")
    out = args.out or REPO / "data" / "pretraining" / "crosscheck" / args.run.name
    files = rollout.choose_clips(args.motion_dir, args.clips, args.seed, args.include, args.clips_file)
    out.mkdir(parents=True, exist_ok=True)
    (out / "clips.txt").write_text("".join(f"{f.stem}\n" for f in files))
    device = getattr(args, "device", None) or "cuda:0"
    actors = load_actors(args.run, device)
    history_length = next(iter(actors.values())).history_length
    env = rollout.make_env(files, args.num_envs or len(files), device, args.seed,
                           task=rollout.play_task(args.run))  # the run's own plant
    try:
        rows = rollout.rollout(env, actors, history_length, args.seconds)
    finally:
        env.close()
    rollout.write_rows(rows, out / "crosscheck.csv")
    line = rollout.summary(str(args.run), rows)
    (out / "summary.txt").write_text(line + "\n")
    print(f"[crosscheck] {line}")
    print(f"[crosscheck] {out / 'crosscheck.csv'}")
    return line


def main() -> None:
    from isaaclab.app import AppLauncher

    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    args.headless = True
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
