import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple


SCRIPT_MECHANISM = "plot_mechanism_abcd_results.py"
SCRIPT_GLOBAL_TASKWISE = "plot_global_acc_all_tasks_by_experiment.py"
SCRIPT_LOCAL_MEAN = "plot_local_mean_acc_by_experiment.py"


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Unified plotting entrypoint for A/B/C/D mechanism experiments. "
            "Runs the three plotting scripts and organizes outputs in one root folder."
        )
    )
    p.add_argument(
        "--runs_root",
        type=str,
        default="",
        help="Experiment output root (e.g., runs_mechanism_abcd_permuted_mnist). If omitted, auto-detect latest runs_mechanism_abcd* folder.",
    )
    p.add_argument(
        "--output_root",
        type=str,
        default="",
        help="Root folder for combined plotting outputs. Default: <runs_root>/figures_bundle",
    )
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title_suffix", type=str, default="")
    p.add_argument("--rolling", type=int, default=1)
    p.add_argument("--skip_mechanism", action="store_true")
    p.add_argument("--skip_global_taskwise", action="store_true")
    p.add_argument("--skip_local_mean", action="store_true")
    p.add_argument(
        "--local_client_weighted",
        action="store_true",
        help="Pass --client_weighted to plot_local_mean_acc_by_experiment.py",
    )
    p.add_argument(
        "--local_overlay_global_baseline",
        action="store_true",
        help="Pass --also_plot_global_local_baseline to plot_local_mean_acc_by_experiment.py",
    )
    p.add_argument(
        "--shock_window_rounds",
        type=int,
        default=10,
        help="Forwarded to plot_mechanism_abcd_results.py",
    )
    p.add_argument(
        "--scatter_window_rounds",
        type=int,
        default=5,
        help="Forwarded to plot_mechanism_abcd_results.py",
    )
    p.add_argument(
        "--progress_bins",
        type=int,
        default=30,
        help="Forwarded to plot_mechanism_abcd_results.py",
    )
    return p.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def auto_detect_latest_runs_root(base: Path) -> Path:
    candidates = [p for p in base.iterdir() if p.is_dir() and p.name.startswith("runs_mechanism_abcd")]
    if not candidates:
        raise FileNotFoundError("No runs_mechanism_abcd* directories found in repo root.")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def run_cmd(cmd: List[str], cwd: Path) -> Tuple[int, str, str]:
    proc = subprocess.run(cmd, cwd=str(cwd), capture_output=True, text=True)
    return proc.returncode, proc.stdout, proc.stderr


def main():
    args = parse_args()
    root = repo_root()
    runs_root = (root / args.runs_root).resolve() if args.runs_root else auto_detect_latest_runs_root(root).resolve()
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    output_root = (root / args.output_root).resolve() if args.output_root else (runs_root / "figures_bundle").resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    script_dir = Path(__file__).resolve().parent
    tasks = []

    if not args.skip_mechanism:
        tasks.append(
            (
                "mechanism_core",
                [
                    sys.executable,
                    str(script_dir / SCRIPT_MECHANISM),
                    "--runs_root",
                    str(runs_root),
                    "--output_dir",
                    str(output_root / "mechanism_core"),
                    "--dpi",
                    str(args.dpi),
                    "--title_suffix",
                    args.title_suffix,
                    "--rolling",
                    str(args.rolling),
                    "--shock_window_rounds",
                    str(args.shock_window_rounds),
                    "--scatter_window_rounds",
                    str(args.scatter_window_rounds),
                    "--progress_bins",
                    str(args.progress_bins),
                ],
            )
        )

    if not args.skip_global_taskwise:
        tasks.append(
            (
                "global_taskwise",
                [
                    sys.executable,
                    str(script_dir / SCRIPT_GLOBAL_TASKWISE),
                    "--runs_root",
                    str(runs_root),
                    "--output_dir",
                    str(output_root / "global_taskwise"),
                    "--dpi",
                    str(args.dpi),
                    "--title_suffix",
                    args.title_suffix,
                    "--rolling",
                    str(args.rolling),
                    "--include_mean",
                ],
            )
        )

    if not args.skip_local_mean:
        cmd = [
            sys.executable,
            str(script_dir / SCRIPT_LOCAL_MEAN),
            "--runs_root",
            str(runs_root),
            "--output_dir",
            str(output_root / "local_mean_old_tasks"),
            "--dpi",
            str(args.dpi),
            "--title_suffix",
            args.title_suffix,
            "--rolling",
            str(args.rolling),
            "--include_mean",
        ]
        if args.local_client_weighted:
            cmd.append("--client_weighted")
        if args.local_overlay_global_baseline:
            cmd.append("--also_plot_global_local_baseline")
        tasks.append(("local_mean_old_tasks", cmd))

    if not tasks:
        raise ValueError("Nothing to run: all plotting groups are skipped.")

    run_log = []
    print(f"runs_root: {runs_root}")
    print(f"output_root: {output_root}")

    for name, cmd in tasks:
        print(f"\n[run] {name}")
        print(" ".join(cmd))
        code, stdout, stderr = run_cmd(cmd, cwd=root)
        if stdout.strip():
            print(stdout.rstrip())
        if stderr.strip():
            print(stderr.rstrip(), file=sys.stderr)
        run_log.append(
            {
                "name": name,
                "command": cmd,
                "returncode": code,
                "stdout_tail": stdout.splitlines()[-20:],
                "stderr_tail": stderr.splitlines()[-20:],
            }
        )
        if code != 0:
            raise RuntimeError(f"Plot task failed: {name} (exit={code})")

    manifest = {
        "runs_root": str(runs_root),
        "output_root": str(output_root),
        "tasks": [t[0] for t in tasks],
        "config": {
            "dpi": args.dpi,
            "title_suffix": args.title_suffix,
            "rolling": args.rolling,
            "shock_window_rounds": args.shock_window_rounds,
            "scatter_window_rounds": args.scatter_window_rounds,
            "progress_bins": args.progress_bins,
            "local_client_weighted": bool(args.local_client_weighted),
            "local_overlay_global_baseline": bool(args.local_overlay_global_baseline),
        },
        "notes": [
            "mechanism_core: 7 figures from plot_mechanism_abcd_results.py",
            "global_taskwise: per-experiment global model task-wise accuracy curves",
            "local_mean_old_tasks: per-experiment local personalized old-task accuracy curves from pers_metrics.csv",
            "If you want local task-wise (current+old) curves from local_eval_metrics.csv, add a new plot script based on local_eval_metrics.",
        ],
        "run_log": run_log,
    }
    (output_root / "bundle_manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[done]")
    print(f"Bundle output: {output_root}")
    for sub in [t[0] for t in tasks]:
        print(f" - {sub}")


if __name__ == "__main__":
    main()
