import argparse
import copy
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np


def _load_npz_payload(npz_file: Path) -> dict:
    with np.load(npz_file, allow_pickle=True) as data:
        payload = data["data"].tolist()
    return payload


def _save_npz_payload(npz_file: Path, payload: dict):
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_file, "wb") as f:
        np.savez_compressed(f, data=payload)


def _resolve_dataset_root(raw_root: str) -> Path:
    p = Path(raw_root)
    if p.is_absolute():
        return p
    # Prefer path relative to repo root when called from repo root; fallback to script dir semantics.
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root_candidate = (Path(__file__).resolve().parents[1] / p).resolve()
    return repo_root_candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create class-disjoint continual-learning tasks from an existing CIFAR100 federated "
            "dataset (NPZ client format). Preserves client non-IID by filtering each client's data."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--source_dataset", type=str, default="cifar100-dir0.1-npz")
    parser.add_argument("--num_tasks", type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=100)
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="cifar100-cl4-task",
        help="Output datasets will be <task_prefix><idx>-npz (1-based idx).",
    )
    parser.add_argument(
        "--shuffle_classes",
        action="store_true",
        help="Shuffle class order before splitting into tasks (deterministic with --seed).",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--require_non_iid",
        action="store_true",
        help="Error if source config.json does not indicate non_iid=true.",
    )
    parser.add_argument(
        "--require_all_clients_nonempty",
        action="store_true",
        help="Error if any task has a client with zero train samples after filtering.",
    )
    parser.add_argument(
        "--relabel_within_task",
        action="store_true",
        help=(
            "Remap labels within each task to [0..task_classes-1]. "
            "Default keeps original CIFAR100 labels (recommended for a 100-way classifier)."
        ),
    )
    parser.add_argument("--dry_run", action="store_true", help="Only print planned split/statistics; do not write files.")
    return parser.parse_args()


def split_classes(num_classes: int, num_tasks: int, shuffle: bool, seed: int) -> List[List[int]]:
    if num_tasks < 1:
        raise ValueError("num_tasks must be >= 1")
    classes = list(range(num_classes))
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(classes)

    chunks: List[List[int]] = []
    base = num_classes // num_tasks
    rem = num_classes % num_tasks
    start = 0
    for t in range(num_tasks):
        size = base + (1 if t < rem else 0)
        chunks.append(classes[start : start + size])
        start += size
    if sum(len(c) for c in chunks) != num_classes:
        raise RuntimeError("Class split failed to cover all classes.")
    return chunks


def filter_payload_by_classes(payload: dict, class_set: Sequence[int], relabel: bool = False) -> dict:
    x = np.array(payload["x"])
    y = np.array(payload["y"])
    if y.size == 0:
        return {"x": x.copy(), "y": y.copy()}

    class_arr = np.array(sorted(class_set), dtype=y.dtype)
    mask = np.isin(y, class_arr)
    x_out = x[mask]
    y_out = y[mask]

    if relabel and y_out.size > 0:
        mapping = {int(c): i for i, c in enumerate(sorted(class_set))}
        y_out = np.array([mapping[int(v)] for v in y_out], dtype=np.int64)

    if x_out.dtype != x.dtype:
        x_out = x_out.astype(x.dtype, copy=False)
    if y_out.dtype != y.dtype and not relabel:
        y_out = y_out.astype(y.dtype, copy=False)
    return {"x": x_out, "y": y_out}


def load_source_config(source_dir: Path) -> dict:
    cfg_path = source_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def collect_client_files(source_dir: Path) -> Tuple[List[Path], List[Path]]:
    train_files = sorted((source_dir / "train").glob("*.npz"))
    test_files = sorted((source_dir / "test").glob("*.npz"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Missing train/test .npz files under {source_dir}")
    return train_files, test_files


def summarize_task_client_counts(
    source_dir: Path,
    task_classes: List[List[int]],
    train_files: List[Path],
    test_files: List[Path],
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]]]:
    task_stats: List[Dict[str, object]] = []
    per_client_rows: List[Dict[str, object]] = []

    for task_idx, cls in enumerate(task_classes, start=1):
        cls_set = set(cls)
        train_counts: List[int] = []
        test_counts: List[int] = []
        unique_labels_in_task = set()

        for train_file, test_file in zip(train_files, test_files):
            cid = int(train_file.stem)
            train_payload = _load_npz_payload(train_file)
            test_payload = _load_npz_payload(test_file)
            y_train = np.array(train_payload["y"])
            y_test = np.array(test_payload["y"])

            tr_mask = np.isin(y_train, list(cls_set))
            te_mask = np.isin(y_test, list(cls_set))
            tr_n = int(np.sum(tr_mask))
            te_n = int(np.sum(te_mask))

            train_counts.append(tr_n)
            test_counts.append(te_n)
            if tr_n > 0:
                unique_labels_in_task.update(np.unique(y_train[tr_mask]).astype(int).tolist())
            if te_n > 0:
                unique_labels_in_task.update(np.unique(y_test[te_mask]).astype(int).tolist())

            per_client_rows.append(
                {
                    "task_idx": task_idx,
                    "client_id": cid,
                    "train_samples": tr_n,
                    "test_samples": te_n,
                }
            )

        task_stats.append(
            {
                "task_idx": task_idx,
                "classes": [int(c) for c in cls],
                "num_classes": len(cls),
                "train_total": int(np.sum(train_counts)),
                "test_total": int(np.sum(test_counts)),
                "train_min_client": int(np.min(train_counts)),
                "train_max_client": int(np.max(train_counts)),
                "train_mean_client": float(np.mean(train_counts)),
                "test_min_client": int(np.min(test_counts)),
                "test_max_client": int(np.max(test_counts)),
                "test_mean_client": float(np.mean(test_counts)),
                "num_empty_train_clients": int(np.sum(np.array(train_counts) == 0)),
                "num_empty_test_clients": int(np.sum(np.array(test_counts) == 0)),
                "labels_present_count": len(unique_labels_in_task),
            }
        )

    return task_stats, per_client_rows


def write_task_dataset(
    source_dir: Path,
    output_dir: Path,
    task_idx: int,
    task_classes: Sequence[int],
    *,
    source_config: dict,
    relabel_within_task: bool,
    train_files: List[Path],
    test_files: List[Path],
):
    cls_set = set(int(c) for c in task_classes)
    for train_file, test_file in zip(train_files, test_files):
        train_payload = _load_npz_payload(train_file)
        test_payload = _load_npz_payload(test_file)

        tr_out = filter_payload_by_classes(train_payload, cls_set, relabel=relabel_within_task)
        te_out = filter_payload_by_classes(test_payload, cls_set, relabel=relabel_within_task)

        _save_npz_payload(output_dir / "train" / train_file.name, tr_out)
        _save_npz_payload(output_dir / "test" / test_file.name, te_out)

    cfg = copy.deepcopy(source_config)
    cfg["cl_mode"] = "class_split"
    cfg["cl_dataset"] = "CIFAR100"
    cfg["cl_task_id"] = int(task_idx)
    cfg["cl_num_tasks"] = None  # set by caller after task count known if needed
    cfg["cl_task_classes"] = [int(c) for c in task_classes]
    cfg["cl_task_num_classes"] = int(len(task_classes))
    cfg["cl_task_classes_disjoint"] = True
    cfg["cl_relabel_within_task"] = bool(relabel_within_task)
    cfg["cl_label_space"] = "task-local" if relabel_within_task else "global-cifar100"
    (output_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=True, separators=(",", ":")),
        encoding="utf-8",
    )


def main():
    args = parse_args()
    dataset_root = _resolve_dataset_root(args.dataset_root)
    source_dir = dataset_root / args.source_dataset
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    source_cfg = load_source_config(source_dir)
    if args.require_non_iid and not bool(source_cfg.get("non_iid", False)):
        raise ValueError(
            f"Source dataset config does not indicate non_iid=true: {source_dir / 'config.json'}"
        )

    train_files, test_files = collect_client_files(source_dir)
    if len(train_files) != len(test_files):
        raise RuntimeError("Train/test client file counts do not match.")

    task_classes = split_classes(
        num_classes=args.num_classes,
        num_tasks=args.num_tasks,
        shuffle=args.shuffle_classes,
        seed=args.seed,
    )

    # Validate class partition is disjoint and complete.
    flat = [c for chunk in task_classes for c in chunk]
    if len(flat) != len(set(flat)):
        raise RuntimeError("Task class sets overlap.")
    if sorted(flat) != list(range(args.num_classes)):
        raise RuntimeError("Task class sets do not cover the full class range.")

    task_stats, per_client_rows = summarize_task_client_counts(
        source_dir=source_dir,
        task_classes=task_classes,
        train_files=train_files,
        test_files=test_files,
    )

    print(f"Source dataset: {source_dir}")
    print(f"Detected clients: {len(train_files)}")
    print(f"Source non_iid flag: {source_cfg.get('non_iid', 'unknown')}")
    print(f"Split mode: class-disjoint ({args.num_tasks} tasks)")
    print(f"Relabel within task: {args.relabel_within_task}")
    for ts in task_stats:
        cls_list = ts["classes"]
        cls_preview = ",".join(str(c) for c in cls_list[:5])
        if len(cls_list) > 5:
            cls_preview += ",..."
        print(
            f"Task {ts['task_idx']}: classes(min={min(cls_list)}, max={max(cls_list)}, sample=[{cls_preview}]) "
            f"(n={ts['num_classes']}) train_total={ts['train_total']} test_total={ts['test_total']} "
            f"empty_train_clients={ts['num_empty_train_clients']} empty_test_clients={ts['num_empty_test_clients']}"
        )

    if args.require_all_clients_nonempty:
        bad = [
            ts
            for ts in task_stats
            if ts["num_empty_train_clients"] > 0 or ts["num_empty_test_clients"] > 0
        ]
        if bad:
            raise ValueError(
                "Some tasks contain empty clients after filtering. "
                "Disable --require_all_clients_nonempty or use a different source partition."
            )

    summary = {
        "source_dataset": args.source_dataset,
        "source_dir": str(source_dir),
        "num_tasks": int(args.num_tasks),
        "num_classes_total": int(args.num_classes),
        "num_clients": len(train_files),
        "shuffle_classes": bool(args.shuffle_classes),
        "seed": int(args.seed),
        "relabel_within_task": bool(args.relabel_within_task),
        "preserved_source_non_iid": bool(source_cfg.get("non_iid", False)),
        "task_stats": task_stats,
        "per_client_stats": per_client_rows,
    }

    summary_path = dataset_root / f"{args.task_prefix.rstrip('-_')}_summary.json"
    if args.dry_run:
        print(f"[dry-run] Summary would be written to: {summary_path}")
        return

    output_dirs: List[Path] = []
    for task_idx, cls in enumerate(task_classes, start=1):
        out_name = f"{args.task_prefix}{task_idx}-npz"
        out_dir = dataset_root / out_name
        output_dirs.append(out_dir)
        print(f"Writing task {task_idx} -> {out_dir}")
        write_task_dataset(
            source_dir=source_dir,
            output_dir=out_dir,
            task_idx=task_idx,
            task_classes=cls,
            source_config=source_cfg,
            relabel_within_task=args.relabel_within_task,
            train_files=train_files,
            test_files=test_files,
        )

    # Patch per-task config with total task count and summary path.
    for out_dir in output_dirs:
        cfg_path = out_dir / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["cl_num_tasks"] = int(args.num_tasks)
        cfg["cl_tasks_summary"] = summary_path.name
        cfg_path.write_text(json.dumps(cfg, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
