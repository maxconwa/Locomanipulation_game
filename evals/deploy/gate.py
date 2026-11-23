"""The deploy gate: is an export ready for the robot? Runs every deploy check, judges each against gate.yaml, and
writes one report: PASS only if every check ran and passed (a check that could not run is NOT RUN, and fails).

    python -m evals.deploy.gate --checkpoint <run>/checkpoints/agent_<N>.pt [--skip golem,parity,...] [--only ...]
    python -m evals.deploy.gate --export <export dir> --out <dir>         (no parity: it needs the checkpoint)

Checks (gate.yaml has their thresholds):
  metric_code  test_sportmode_metrics.py: the scorecards' metric code reproduces the sport-mode report (CPU)
  parity       parity.py: the deploy controller against the training env, step by step (Isaac, GPU; --isaac_python)
  stress       stress_mujoco.py: the hardware-perturbation grid (no falls in its zero-fall cells), the 180 s stands,
               a small command after a 60 s stand (no falls), the pre-pose engage grid (every cell passes) (CPU)
  bag_replay   bag_replay.py: the robot's bags through lmg_deploy's controller: the deployed policy's replayed
               targets match the recorded ones; the candidate's LSTM stays in its simulated range, no sim falls (CPU)
  golem        golem/run.sh: the export under lmg_deploy in GOLEM's RoboCasa sim, THE command program (docker, GPU)

Uses ~/GOLEM as checked out ($GOLEM); the report records its commit. Writes logs/evals/<run>/deploy/gate_<N>/
(report.md, report.json, each check's own folder and log); exit 0 on PASS, 1 otherwise. Run from the repo root with
a python that has mujoco, torch, numpy, scipy, pyyaml, pin and rosbags (ROSBAGS_PATH is put on PYTHONPATH if set).
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))  # the repo root
from evals.common import conditions, outputs, runs  # noqa: E402

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CHECKS = ("metric_code", "parity", "stress", "bag_replay", "golem")
PASS, FAIL, NOT_RUN = "PASS", "FAIL", "NOT RUN"


def run(cmd, log: Path, env=None, cwd=REPO) -> int:
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[gate] {' '.join(str(c) for c in cmd)}  > {log}", flush=True)
    with open(log, "w") as f:
        return subprocess.call([str(c) for c in cmd], stdout=f, stderr=subprocess.STDOUT, cwd=str(cwd),
                               env={**os.environ, **(env or {})})


def tail(log: Path, n=3) -> str:
    """The log's last error line (an exception's message), else its last n lines."""
    try:
        lines = open(log, errors="replace").read().strip().splitlines()
    except OSError:
        return "(no log)"
    errors = [ln.strip() for ln in lines if ln.lstrip().split(":")[0].endswith(("Error", "Exception", "SystemExit"))]
    return errors[-1][:400] if errors else " / ".join(ln.strip() for ln in lines[-n:])


def check_metric_code(a, g, d: Path):
    log = d / "metric_code.log"
    code = run([sys.executable, "-m", "evals.deploy.test_sportmode_metrics"], log)
    text = open(log, errors="replace").read()
    if "FileNotFoundError" in text or "no data/" in text:
        return NOT_RUN, "the sport-mode analysis dir (bags) is not on this machine", {}
    return (PASS if code == 0 else FAIL), tail(log, 1), {}


def check_parity(a, g, d: Path, export: Path):
    if a.checkpoint is None:
        return NOT_RUN, "parity needs the checkpoint (--checkpoint)", {}
    out = d / "parity"
    log = d / "parity.log"
    code = run([a.isaac_python, "-m", "evals.deploy.parity", "--checkpoint", a.checkpoint, "--export", export,
                "--deploy_pkg", Path(a.golem) / "core_ws/src/lmg_deploy", "--tolerance", g["parity"]["tolerance"],
                "--out", out], log)
    pj = out / "parity.json"
    if not pj.is_file():
        return NOT_RUN, f"parity did not finish (exit {code}): {tail(log)}", {}
    worst = json.load(open(pj)).get("worst_abs_diff", {})
    over = {k: v for k, v in worst.items() if v > g["parity"]["tolerance"]}
    return (FAIL if over else PASS), (f"over {g['parity']['tolerance']:g}: {over}" if over else
                                      f"worst {max(worst.values(), default=0):.1e}"), {"worst": worst}


def check_stress(a, g, d: Path, export: Path):
    s = g["stress"]
    jobs = {
        "grid": ["--seeds", s["seeds"]],
        "long": ["--conditions", ",".join(s["long_conditions"]), "--profiles", "zero", "--seeds", s["seeds"],
                 "--seconds", 180],
        "small60": ["--conditions", ",".join(s["small60_conditions"]), "--profiles", "small60", "--seeds", s["seeds"],
                    "--seconds", 80],
        "prepose": ["--conditions", "pp", "--plants", "robocasa", "--seeds", s["pp_seeds"]],
    }
    problems, details, missing = [], {}, []
    for name, extra in jobs.items():
        out = d / f"stress_{name}"
        code = run([sys.executable, "-m", "evals.deploy.stress_mujoco", "--export", export, "--label", name,
                    "--out", out, "--workers", a.workers] + extra, d / f"stress_{name}.log")
        f = out / "summary.json"
        if not f.is_file():
            missing.append(f"{name} (exit {code}: {tail(d / f'stress_{name}.log', 1)})")
            continue
        summ = json.load(open(f))
        cells, pp = summ.get("cells", {}), summ.get("pp_cells", {})
        details[name] = {"cells": {k: v.get("fell_pct") for k, v in cells.items()},
                         "pp": {k: v.get("pass") for k, v in pp.items()}}
        if name == "grid":
            for key, v in cells.items():
                if key.split("/")[0] in s["zero_fall_cells"] and (v.get("fell_pct") or 0) > 0:
                    problems.append(f"{key} fell {v['fell_pct']:.0f}%")
        elif name in ("long", "small60"):
            problems += [f"{name} {k} fell {v['fell_pct']:.0f}%" for k, v in cells.items() if (v.get("fell_pct") or 0) > 0]
        else:
            problems += [f"engage {k} failed its marks" for k, v in pp.items() if not v.get("pass")]
    if missing:
        return NOT_RUN, "did not finish: " + "; ".join(missing), details
    return (FAIL if problems else PASS), ("; ".join(problems[:8]) + (" ..." if len(problems) > 8 else "")
                                          if problems else "no falls in the gated cells; every engage cell passed"), details


def check_bag_replay(a, g, d: Path, export: Path):
    b = g["bag_replay"]
    bags_dir = Path(os.path.expanduser(b["bags_dir"]))
    deployed = REPO / b["deployed_export"]
    if not bags_dir.is_dir() or not deployed.is_dir():
        return NOT_RUN, f"missing {bags_dir if not bags_dir.is_dir() else deployed}", {}
    env = {"PYTHONPATH": os.environ.get("ROSBAGS_PATH", "") + os.pathsep + os.environ.get("PYTHONPATH", "")}
    problems, details, missing = [], {}, []
    for bag in b["bags"]:
        out = d / f"replay_{bag}"
        code = run([sys.executable, "-m", "evals.deploy.bag_replay", "--bag", bags_dir / bag, "--deployed",
                    b["deployed"], "--golem", a.golem, "--policy", f"{b['deployed']}={deployed}",
                    "--policy", f"candidate={export}", "--sim_seeds", b["sim_seeds"], "--out", out], d / f"replay_{bag}.log",
                   env=env)
        f = out / "summary.json"
        if not f.is_file():
            missing.append(f"{bag} (exit {code}: {tail(d / f'replay_{bag}.log', 1)})")
            continue
        pol = json.load(open(f))["policies"]
        tc = (pol.get(b["deployed"], {}).get("target_check") or {}).get("all", {})
        cand = pol.get("candidate", {})
        details[bag] = {"target_check": tc, "candidate": {k: cand.get(k, {}).get("steps_outside_sim_c")
                                                           for k in ("legs", "arms")},
                        "sim_falls": cand.get("sim_falls")}
        if tc and (tc["median"] > b["max_target_median"] or tc["p99"] > b["max_target_p99"]):
            problems.append(f"{bag}: the replay is not the node (targets median {tc['median']:.1e}, p99 {tc['p99']:.1e})")
        for actor in ("legs", "arms"):
            v = cand.get(actor, {}).get("steps_outside_sim_c")
            if v is not None and v > b["max_outside_sim_c"]:
                problems.append(f"{bag}: {actor} LSTM outside its sim range {100 * v:.1f}% of steps")
        if any(f is not None for f in (cand.get("sim_falls") or [])):
            problems.append(f"{bag}: fell in MuJoCo on the bag's commands")
    if missing:
        return NOT_RUN, "did not finish: " + "; ".join(missing), details
    return (FAIL if problems else PASS), ("; ".join(problems) if problems else "matches the node; in range"), details


def check_golem(a, g, d: Path, export: Path):
    out = d / "golem"
    log = d / "golem.log"
    code = run(["bash", HERE / "golem" / "run.sh", export, out], log,
               env={"GOLEM": a.golem, "SCORE_PYTHON": sys.executable})
    sj = out / "score.json"
    if not sj.is_file():
        return NOT_RUN, f"the GOLEM run did not reach its scoring (exit {code}): {tail(log)}", {}
    s = json.load(open(sj))
    failed = [c for c in s["checks"] if not c["pass"]]
    return (PASS if s["passed"] else FAIL), ("; ".join(f"{c['check']}: {c['detail']}" for c in failed)[:600]
                                            if failed else "every check passed"), {"checks": s["checks"],
                                                                                   "moves": s.get("moves")}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--export")
    ap.add_argument("--out")
    ap.add_argument("--skip", default="", help="Comma list of checks to leave out (reported as skipped, not a pass).")
    ap.add_argument("--only", default="", help="Comma list: run only these.")
    ap.add_argument("--gate", default=str(HERE / "gate.yaml"))
    ap.add_argument("--golem", default=os.environ.get("GOLEM", os.path.expanduser("~/GOLEM")))
    ap.add_argument("--isaac_python", default=os.environ.get(
        "ISAAC_PYTHON", os.path.expanduser("~/anaconda3/envs/env_isaaclab/bin/python")))
    ap.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 1))
    a = ap.parse_args()
    if not (a.checkpoint or a.export):
        ap.error("--checkpoint or --export")
    g = yaml.safe_load(open(a.gate))
    from evals.mujoco.policy import export_for
    if a.checkpoint:
        a.checkpoint = os.path.abspath(a.checkpoint)
        run_dir = runs.run_dir_of(a.checkpoint)
        export = Path(a.export).resolve() if a.export else export_for(
            a.checkpoint, outputs.eval_dir(runs.name(run_dir), "mujoco", "exports"))
        d = Path(a.out) if a.out else outputs.eval_dir(runs.name(run_dir), "deploy", f"gate_{runs.step_of(a.checkpoint)}")
    else:
        export = Path(a.export).resolve()
        if not a.out:
            ap.error("--out with --export")
        d = Path(a.out)
    d.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CHECKS if (not a.only or c in a.only.split(",")) and c not in a.skip.split(",")]
    fns = {"metric_code": check_metric_code, "parity": check_parity, "stress": check_stress,
           "bag_replay": check_bag_replay, "golem": check_golem}
    results = {}
    for name in CHECKS:
        if name not in selected:
            results[name] = {"status": "skipped", "detail": "--skip / --only"}
            continue
        try:
            args = (a, g, d) if name == "metric_code" else (a, g, d, export)
            status, detail, extra = fns[name](*args)
        except Exception as e:  # noqa: BLE001  (a check's own failure to run is NOT RUN, with why)
            status, detail, extra = NOT_RUN, f"{type(e).__name__}: {e}", {}
        results[name] = {"status": status, "detail": detail, **extra}
        print(f"[gate] {name}: {status} — {detail}", flush=True)
    ran = [r["status"] for r in results.values() if r["status"] != "skipped"]
    verdict = PASS if ran and all(s == PASS for s in ran) else FAIL
    golem_commit = subprocess.run(["git", "-C", a.golem, "rev-parse", "--short", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
    golem_branch = subprocess.run(["git", "-C", a.golem, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True,
                                  text=True).stdout.strip()
    report = {"verdict": verdict, "checkpoint": a.checkpoint, "export": str(export),
              "time": datetime.datetime.now().isoformat(timespec="seconds"), "code_commit": conditions.commit(),
              "golem": f"{golem_branch} {golem_commit}", "skipped": [c for c in CHECKS if c not in selected],
              "checks": results, "gate": g}
    json.dump(report, open(d / "report.json", "w"), indent=1, default=str)
    lines = [f"# Deploy gate: {verdict}", "", f"Export `{export}`" + (f", checkpoint `{a.checkpoint}`" if a.checkpoint else ""),
             f"GOLEM {report['golem']}; code {report['code_commit']}; {report['time']}.",
             "" if not report["skipped"] else f"**Skipped (not judged): {', '.join(report['skipped'])}.**", "",
             "| Check | Result | Detail |", "| --- | --- | --- |"]
    lines += [f"| {k} | {'**' + v['status'] + '**' if v['status'] != PASS else v['status']} | {v['detail']} |"
              for k, v in results.items()]
    lines += ["", "Thresholds: evals/deploy/gate.yaml. Each check's output is in its folder beside this report."]
    open(d / "report.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if verdict == PASS else 1


if __name__ == "__main__":
    sys.exit(main())
