import argparse
import copy
import csv
import json
import os
import random
import statistics
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from flcore.servers.serverALA import FedALA
from flcore.trainmodel.models import FedAvgCNN, fastText
from utils.data_utils import read_client_data


# Keep the same constants as main.py for fastText.
VOCAB_SIZE = 98635
HIDDEN_DIM = 32


@dataclass
class ContinualResult:
    run_id: int
    seed: int
    task1_acc_before: float
    task1_acc_after_task1: float
    task1_peak_during_task1: float
    task2_acc_before: float
    task2_acc_after_task2: float
    task1_acc_after_task2: float
    forgetting_from_switch: float
    forgetting_from_peak: float
    task2_gain: float


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(model_name: str, dataset_name: str, num_classes: int, device: str):
    dataset_name = dataset_name.lower()
    if model_name == "cnn":
        if "mnist" in dataset_name:
            model = FedAvgCNN(in_features=1, num_classes=num_classes, dim=1024)
        elif "cifar" in dataset_name:
            model = FedAvgCNN(in_features=3, num_classes=num_classes, dim=1600)
        else:
            model = FedAvgCNN(in_features=3, num_classes=num_classes, dim=10816)
    elif model_name == "resnet":
        model = torchvision.models.resnet18(pretrained=False, num_classes=num_classes)
    elif model_name == "fastText":
        model = fastText(hidden_dim=HIDDEN_DIM, vocab_size=VOCAB_SIZE, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return model.to(device)


def build_args(
    dataset: str,
    model: torch.nn.Module,
    device: str,
    num_classes: int,
    rounds: int,
    eval_gap: int,
    local_lr: float,
    local_steps: int,
    batch_size: int,
    num_clients: int,
    join_ratio: float,
    eta: float,
    rand_percent: int,
    layer_idx: int,
    ala_threshold: float,
    ala_num_pre_loss: int,
):
    return SimpleNamespace(
        device=device,
        dataset=dataset,
        global_rounds=rounds,
        model=model,
        num_clients=num_clients,
        join_ratio=join_ratio,
        random_join_ratio=False,
        eval_gap=eval_gap,
        num_classes=num_classes,
        batch_size=batch_size,
        local_learning_rate=local_lr,
        local_steps=local_steps,
        eta=eta,
        rand_percent=rand_percent,
        layer_idx=layer_idx,
        ala_threshold=ala_threshold,
        ala_num_pre_loss=ala_num_pre_loss,
    )


def evaluate_on_dataset(
    model: torch.nn.Module,
    dataset: str,
    num_clients: int,
    batch_size: int,
    device: str,
) -> float:
    model.eval()
    total_correct = 0
    total_count = 0

    with torch.no_grad():
        for cid in range(num_clients):
            test_data = read_client_data(dataset, cid, is_train=False)
            loader = DataLoader(test_data, batch_size=batch_size, drop_last=False, shuffle=False)
            for x, y in loader:
                if isinstance(x, list):
                    x[0] = x[0].to(device)
                elif isinstance(x, tuple):
                    x = (x[0].to(device), x[1].to(device))
                else:
                    x = x.to(device)
                y = y.to(device)
                logits = model(x)
                total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
                total_count += y.shape[0]

    if total_count == 0:
        raise ValueError(f"No test samples found for dataset {dataset}")
    return total_correct / total_count


def summarize(results, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = [r.__dict__ for r in results]
    csv_path = output_dir / "continual_switch_runs.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    metric_names = [
        "task1_acc_before",
        "task1_acc_after_task1",
        "task1_peak_during_task1",
        "task2_acc_before",
        "task2_acc_after_task2",
        "task1_acc_after_task2",
        "forgetting_from_switch",
        "forgetting_from_peak",
        "task2_gain",
    ]
    summary = {}
    for name in metric_names:
        vals = [getattr(r, name) for r in results]
        summary[name] = {
            "mean": float(statistics.fmean(vals)),
            "std": float(statistics.pstdev(vals) if len(vals) > 1 else 0.0),
        }

    json_path = output_dir / "continual_switch_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    return csv_path, json_path, summary


def main():
    parser = argparse.ArgumentParser(
        description="Run FedALA with task1->task2 switching and report forgetting/accuracy."
    )
    parser.add_argument("--task1_dataset", type=str, required=True)
    parser.add_argument("--task2_dataset", type=str, required=True)
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet", "fastText"])
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--rounds_task1", type=int, default=200)
    parser.add_argument("--rounds_task2", type=int, default=200)
    parser.add_argument("--times", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--local_lr_task1", type=float, default=0.1)
    parser.add_argument("--local_lr_task2", type=float, default=0.1)
    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--join_ratio", type=float, default=1.0)
    parser.add_argument("--eval_gap", type=int, default=20)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rand_percent", type=int, default=80)
    parser.add_argument("--layer_idx", type=int, default=1)
    parser.add_argument("--ala_threshold", type=float, default=0.01)
    parser.add_argument("--ala_num_pre_loss", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--device_id", type=str, default="0")
    parser.add_argument("--output_dir", type=str, default="./continual_logs")
    args = parser.parse_args()

    if args.rounds_task1 < 1 or args.rounds_task2 < 1:
        raise ValueError("rounds_task1 and rounds_task2 must be >= 1.")
    if args.times < 1:
        raise ValueError("times must be >= 1.")

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        args.device = "cpu"

    results = []
    for run_id in range(args.times):
        run_seed = args.seed + run_id
        set_random_seed(run_seed)
        print(f"\n===== Run {run_id + 1}/{args.times} | seed={run_seed} =====")

        base_model = build_model(args.model, args.task1_dataset, args.num_classes, args.device)

        # Evaluate random initialization on task 1 before any training.
        task1_acc_before = evaluate_on_dataset(
            base_model, args.task1_dataset, args.num_clients, args.batch_size, args.device
        )
        print(f"Task1 acc before training: {task1_acc_before:.4f}")

        # Train on task 1.
        args_t1 = build_args(
            dataset=args.task1_dataset,
            model=copy.deepcopy(base_model),
            device=args.device,
            num_classes=args.num_classes,
            rounds=args.rounds_task1,
            eval_gap=args.eval_gap,
            local_lr=args.local_lr_task1,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            num_clients=args.num_clients,
            join_ratio=args.join_ratio,
            eta=args.eta,
            rand_percent=args.rand_percent,
            layer_idx=args.layer_idx,
            ala_threshold=args.ala_threshold,
            ala_num_pre_loss=args.ala_num_pre_loss,
        )
        server_t1 = FedALA(args_t1, run_id)
        server_t1.train()
        model_after_t1 = copy.deepcopy(server_t1.global_model).to(args.device)

        task1_acc_after_t1 = evaluate_on_dataset(
            model_after_t1, args.task1_dataset, args.num_clients, args.batch_size, args.device
        )
        task1_peak = max(server_t1.rs_test_acc) if server_t1.rs_test_acc else task1_acc_after_t1
        task2_acc_before = evaluate_on_dataset(
            model_after_t1, args.task2_dataset, args.num_clients, args.batch_size, args.device
        )
        print(f"Task1 acc after task1 training: {task1_acc_after_t1:.4f}")
        print(f"Task2 acc before task2 training: {task2_acc_before:.4f}")

        # Train on task 2, initialized by task-1 model.
        args_t2 = build_args(
            dataset=args.task2_dataset,
            model=copy.deepcopy(model_after_t1),
            device=args.device,
            num_classes=args.num_classes,
            rounds=args.rounds_task2,
            eval_gap=args.eval_gap,
            local_lr=args.local_lr_task2,
            local_steps=args.local_steps,
            batch_size=args.batch_size,
            num_clients=args.num_clients,
            join_ratio=args.join_ratio,
            eta=args.eta,
            rand_percent=args.rand_percent,
            layer_idx=args.layer_idx,
            ala_threshold=args.ala_threshold,
            ala_num_pre_loss=args.ala_num_pre_loss,
        )
        server_t2 = FedALA(args_t2, run_id)
        server_t2.train()
        model_after_t2 = copy.deepcopy(server_t2.global_model).to(args.device)

        task2_acc_after_t2 = evaluate_on_dataset(
            model_after_t2, args.task2_dataset, args.num_clients, args.batch_size, args.device
        )
        task1_acc_after_t2 = evaluate_on_dataset(
            model_after_t2, args.task1_dataset, args.num_clients, args.batch_size, args.device
        )

        forgetting_switch = task1_acc_after_t1 - task1_acc_after_t2
        forgetting_peak = task1_peak - task1_acc_after_t2
        task2_gain = task2_acc_after_t2 - task2_acc_before

        print(f"Task2 acc after task2 training: {task2_acc_after_t2:.4f}")
        print(f"Task1 acc after switching to task2: {task1_acc_after_t2:.4f}")
        print(f"Forgetting (from switch point): {forgetting_switch:.4f}")
        print(f"Forgetting (from peak): {forgetting_peak:.4f}")

        results.append(
            ContinualResult(
                run_id=run_id,
                seed=run_seed,
                task1_acc_before=task1_acc_before,
                task1_acc_after_task1=task1_acc_after_t1,
                task1_peak_during_task1=task1_peak,
                task2_acc_before=task2_acc_before,
                task2_acc_after_task2=task2_acc_after_t2,
                task1_acc_after_task2=task1_acc_after_t2,
                forgetting_from_switch=forgetting_switch,
                forgetting_from_peak=forgetting_peak,
                task2_gain=task2_gain,
            )
        )

    csv_path, json_path, summary = summarize(results, Path(args.output_dir).resolve())
    print("\n===== Summary (mean +/- std) =====")
    for key, stat in summary.items():
        print(f"{key}: {stat['mean']:.4f} +/- {stat['std']:.4f}")
    print(f"\nPer-run results: {csv_path}")
    print(f"Summary: {json_path}")


if __name__ == "__main__":
    main()

"""
python run_continual_switch.py `
  --task1_dataset permuted-mnist-task1-npz `
  --task2_dataset permuted-mnist-task2-npz `
  --rounds_task1 40 --rounds_task2 40 --times 1 `
  --device cuda --device_id 0 --eval_gap 20 `
  --output_dir ./continual_logs_permuted_40
"""
