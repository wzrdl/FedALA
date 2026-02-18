import argparse
import csv
import re
import shlex
import statistics
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Experiment:
    key: str
    setting: str
    paper_column: str
    family: str
    model: str
    num_classes: int
    local_lr: float


EXPERIMENTS = [
    Experiment("pathological_mnist", "pathological", "MNIST", "mnist", "cnn", 10, 0.1),
    Experiment("pathological_cifar10", "pathological", "Cifar10", "cifar10", "cnn", 10, 0.005),
    Experiment("pathological_cifar100", "pathological", "Cifar100", "cifar100", "cnn", 100, 0.005),
    Experiment("practical_cifar10", "practical", "Cifar10", "cifar10", "cnn", 10, 0.005),
    Experiment("practical_cifar100", "practical", "Cifar100", "cifar100", "cnn", 100, 0.005),
    Experiment("practical_tiny_cnn", "practical", "TINY", "tiny", "cnn", 200, 0.005),
    Experiment("practical_tiny_resnet", "practical", "TINY*", "tiny", "resnet", 200, 0.1),
    Experiment("practical_agnews", "practical", "AG News", "agnews", "fastText", 4, 0.1),
]


def parse_overrides(dataset_args: List[str]) -> Dict[str, str]:
    overrides: Dict[str, str] = {}
    for item in dataset_args:
        if "=" not in item:
            raise ValueError(f"Invalid --dataset value '{item}'. Expected KEY=DATASET_DIR.")
        key, dataset_name = item.split("=", 1)
        key = key.strip()
        dataset_name = dataset_name.strip()
        if not key or not dataset_name:
            raise ValueError(f"Invalid --dataset value '{item}'. Expected KEY=DATASET_DIR.")
        overrides[key] = dataset_name
    return overrides


def required_tokens(family: str) -> List[str]:
    if family == "mnist":
        return ["mnist"]
    if family == "cifar10":
        return ["cifar10"]
    if family == "cifar100":
        return ["cifar100"]
    if family == "tiny":
        return ["tiny"]
    if family == "agnews":
        return ["news"]
    raise ValueError(f"Unsupported family: {family}")


def score_dataset_name(dataset_name: str, exp: Experiment) -> int:
    name = dataset_name.lower()
    tokens = required_tokens(exp.family)
    if not all(token in name for token in tokens):
        return -1

    score = 10 * len(tokens)
    if exp.setting == "pathological":
        if ("path" in name) or ("pat" in name):
            score += 6
        else:
            score -= 1
    else:
        if "0.1" in name:
            score += 6
        if "dir" in name:
            score += 3
        if "default" in name:
            score += 2

    return score


def resolve_dataset_name(
    dataset_names: List[str], exp: Experiment, overrides: Dict[str, str]
) -> Optional[str]:
    if exp.key in overrides:
        return overrides[exp.key]

    scored = []
    for dataset_name in dataset_names:
        score = score_dataset_name(dataset_name, exp)
        if score >= 0:
            scored.append((score, dataset_name))

    if not scored:
        return None

    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score = scored[0][0]
    best_names = [name for score, name in scored if score == best_score]

    # Avoid silently picking the wrong split when multiple names score the same.
    if len(best_names) > 1:
        return None

    return best_names[0]


def parse_best_accuracies(log_text: str) -> List[float]:
    pattern = re.compile(r"Best global accuracy\.\s*[\r\n]+([0-9]*\.?[0-9]+)")
    return [float(match.group(1)) for match in pattern.finditer(log_text)]


def build_cmd(args, exp: Experiment, dataset_name: str) -> List[str]:
    return [
        args.python,
        "-u",
        "main.py",
        "-t",
        str(args.times),
        "-jr",
        str(args.join_ratio),
        "-nc",
        str(args.num_clients),
        "-nb",
        str(exp.num_classes),
        "-data",
        dataset_name,
        "-m",
        exp.model,
        "-algo",
        "FedALA",
        "-et",
        str(args.eta),
        "-p",
        str(args.layer_idx),
        "-s",
        str(args.rand_percent),
        "-did",
        args.device_id,
        "-gr",
        str(args.rounds),
        "-ls",
        str(args.local_steps),
        "-lbs",
        str(args.batch_size),
        "-lr",
        str(exp.local_lr),
        "-eg",
        str(args.eval_gap),
        "-dev",
        args.device,
        "--ala_threshold",
        str(args.ala_threshold),
        "--ala_num_pre_loss",
        str(args.ala_num_pre_loss),
        "--seed",
        str(args.seed),
    ]


def select_experiments(only: str) -> List[Experiment]:
    if not only.strip():
        return EXPERIMENTS

    wanted = {token.strip() for token in only.split(",") if token.strip()}
    unknown = sorted(wanted - {exp.key for exp in EXPERIMENTS})
    if unknown:
        raise ValueError(
            f"Unknown experiment keys in --only: {unknown}. "
            f"Valid keys: {[exp.key for exp in EXPERIMENTS]}"
        )

    return [exp for exp in EXPERIMENTS if exp.key in wanted]


def main():
    parser = argparse.ArgumentParser(
        description="Run FedALA experiments to reproduce Table 2 settings."
    )
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable.")
    parser.add_argument("--dataset_root", type=str, default="../dataset")
    parser.add_argument("--log_dir", type=str, default="./table2_logs")
    parser.add_argument(
        "--dataset",
        action="append",
        default=[],
        help="Override dataset dir name for a key: KEY=DATASET_DIR",
    )
    parser.add_argument(
        "--only",
        type=str,
        default="",
        help="Comma separated experiment keys to run. Empty means all.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print commands.")
    parser.add_argument(
        "--stop_on_error", action="store_true", help="Stop immediately on first failed run."
    )
    parser.add_argument(
        "--skip_completed",
        action="store_true",
        help="Skip experiments whose log already has parsed results.",
    )

    # Paper defaults for Table 2.
    parser.add_argument("--rounds", type=int, default=2000)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--join_ratio", type=float, default=1.0)
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--eval_gap", type=int, default=1)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rand_percent", type=int, default=80)
    parser.add_argument("--layer_idx", type=int, default=1)
    parser.add_argument("--ala_threshold", type=float, default=0.01)
    parser.add_argument("--ala_num_pre_loss", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--device_id", type=str, default="0")
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    log_dir = Path(args.log_dir).resolve()
    log_dir.mkdir(parents=True, exist_ok=True)

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    dataset_names = sorted([p.name for p in dataset_root.iterdir() if p.is_dir()])
    overrides = parse_overrides(args.dataset)
    selected = select_experiments(args.only)

    assignments: Dict[str, str] = {}
    unresolved: List[str] = []
    for exp in selected:
        dataset_name = resolve_dataset_name(dataset_names, exp, overrides)
        if dataset_name is None:
            unresolved.append(exp.key)
            continue
        if dataset_name not in dataset_names:
            raise FileNotFoundError(
                f"Dataset dir '{dataset_name}' (for {exp.key}) is not under {dataset_root}."
            )
        assignments[exp.key] = dataset_name

    if unresolved:
        print("Could not auto-resolve dataset dirs for keys:")
        for key in unresolved:
            print(f"  - {key}")
        print("\nUse --dataset KEY=DATASET_DIR for each unresolved key.")
        print("Available dataset dirs under dataset_root:")
        for name in dataset_names:
            print(f"  - {name}")
        raise SystemExit(1)

    results = []
    workdir = Path(__file__).resolve().parent

    print("Resolved Table 2 experiments:")
    for exp in selected:
        print(f"  - {exp.key}: {assignments[exp.key]} ({exp.paper_column}, {exp.setting})")

    for exp in selected:
        dataset_name = assignments[exp.key]
        log_file = log_dir / f"{exp.key}.log"
        cmd = build_cmd(args, exp, dataset_name)
        cmd_str = shlex.join(cmd)

        if args.skip_completed and log_file.exists():
            previous = parse_best_accuracies(log_file.read_text(encoding="utf-8", errors="ignore"))
            if previous:
                mean_acc = statistics.fmean(previous)
                std_acc = statistics.pstdev(previous) if len(previous) > 1 else 0.0
                print(
                    f"[SKIP] {exp.key}: found {len(previous)} run(s) in existing log. "
                    f"mean={mean_acc:.4f}, std={std_acc:.4f}"
                )
                results.append(
                    {
                        "key": exp.key,
                        "column": exp.paper_column,
                        "dataset": dataset_name,
                        "status": "skipped",
                        "runs": len(previous),
                        "mean": mean_acc,
                        "std": std_acc,
                        "log": str(log_file),
                    }
                )
                continue

        print(f"\n[RUN] {exp.key}")
        print(f"      {cmd_str}")
        if args.dry_run:
            results.append(
                {
                    "key": exp.key,
                    "column": exp.paper_column,
                    "dataset": dataset_name,
                    "status": "dry_run",
                    "runs": 0,
                    "mean": None,
                    "std": None,
                    "log": str(log_file),
                }
            )
            continue

        with open(log_file, "w", encoding="utf-8") as fout:
            proc = subprocess.run(cmd, cwd=workdir, stdout=fout, stderr=subprocess.STDOUT)

        log_text = log_file.read_text(encoding="utf-8", errors="ignore")
        accuracies = parse_best_accuracies(log_text)

        if proc.returncode != 0:
            print(f"[FAIL] {exp.key}: process exited with code {proc.returncode}. See {log_file}")
            results.append(
                {
                    "key": exp.key,
                    "column": exp.paper_column,
                    "dataset": dataset_name,
                    "status": f"failed({proc.returncode})",
                    "runs": len(accuracies),
                    "mean": None,
                    "std": None,
                    "log": str(log_file),
                }
            )
            if args.stop_on_error:
                break
            continue

        if not accuracies:
            print(f"[FAIL] {exp.key}: no 'Best global accuracy' parsed. See {log_file}")
            results.append(
                {
                    "key": exp.key,
                    "column": exp.paper_column,
                    "dataset": dataset_name,
                    "status": "failed(parse)",
                    "runs": 0,
                    "mean": None,
                    "std": None,
                    "log": str(log_file),
                }
            )
            if args.stop_on_error:
                break
            continue

        mean_acc = statistics.fmean(accuracies)
        std_acc = statistics.pstdev(accuracies) if len(accuracies) > 1 else 0.0
        print(f"[OK] {exp.key}: runs={len(accuracies)}, mean={mean_acc:.4f}, std={std_acc:.4f}")
        results.append(
            {
                "key": exp.key,
                "column": exp.paper_column,
                "dataset": dataset_name,
                "status": "ok",
                "runs": len(accuracies),
                "mean": mean_acc,
                "std": std_acc,
                "log": str(log_file),
            }
        )

    summary_csv = log_dir / "table2_summary.csv"
    with open(summary_csv, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(
            fout, fieldnames=["key", "column", "dataset", "status", "runs", "mean", "std", "log"]
        )
        writer.writeheader()
        writer.writerows(results)

    print("\nSummary:")
    for item in results:
        mean = "-" if item["mean"] is None else f"{item['mean']:.4f}"
        std = "-" if item["std"] is None else f"{item['std']:.4f}"
        print(
            f"  {item['key']:<24} status={item['status']:<12} "
            f"runs={item['runs']:<2} mean={mean:<8} std={std:<8} dataset={item['dataset']}"
        )
    print(f"\nSaved summary CSV: {summary_csv}")


if __name__ == "__main__":
    main()


"""
# PowerShell:
python reproduce_table2.py `
  --dataset pathological_mnist=mnist-pathological-npz `
  --dataset pathological_cifar10=cifar10-pathological-npz `
  --dataset pathological_cifar100=cifar100-pathological-npz `
  --dataset practical_cifar10=cifar10-dir0.1-npz `
  --dataset practical_cifar100=cifar100-dir0.1-npz `
  --dataset practical_tiny_cnn=tiny-imagenet-dir0.1-npz `
  --dataset practical_tiny_resnet=tiny-imagenet-dir0.1-npz `
  --dataset practical_agnews=agnews-dir0.1-npz `
  --device cuda --device_id 0


  python reproduce_table2.py --only practical_tiny_resnet --dataset practical_tiny_resnet=tiny-imagenet-dir0.1-npz --log_dir ./my_tiny_resnet_logs --device cuda --device_id 0
"""