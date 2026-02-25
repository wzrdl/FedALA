import argparse
import copy
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


def _load_npz_payload(npz_file: Path) -> dict:
    with np.load(npz_file, allow_pickle=True) as data:
        return data["data"].tolist()


def _save_npz_payload(npz_file: Path, payload: dict) -> None:
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_file, "wb") as f:
        np.savez_compressed(f, data=payload)


def _resolve_dataset_root(raw_root: str) -> Path:
    p = Path(raw_root)
    if p.is_absolute():
        return p
    cwd_candidate = (Path.cwd() / p).resolve()
    if cwd_candidate.exists():
        return cwd_candidate
    repo_root_candidate = (Path(__file__).resolve().parents[1] / p).resolve()
    return repo_root_candidate


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a multi-task Permuted-MNIST continual-learning dataset from an existing "
            "federated MNIST client NPZ dataset. Client partition (non-IID) is preserved."
        )
    )
    parser.add_argument("--dataset_root", type=str, default="dataset")
    parser.add_argument("--source_dataset", type=str, default="mnist-0.1-npz")
    parser.add_argument("--num_tasks", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=28)
    parser.add_argument(
        "--task_prefix",
        type=str,
        default="permuted-mnist-task",
        help="Output datasets will be <task_prefix><idx>-npz (1-based idx).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Base RNG seed for generating task permutation seeds.")
    parser.add_argument(
        "--explicit_task_seeds",
        type=str,
        default="",
        help=(
            "Optional comma-separated permutation seeds for non-identity tasks. "
            "If task1 is identity (default), provide num_tasks-1 seeds."
        ),
    )
    parser.add_argument(
        "--require_non_iid",
        action="store_true",
        help="Error if source config.json does not indicate non_iid=true.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print planned tasks/statistics; do not write files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into existing output task directories.",
    )
    parser.set_defaults(task1_identity=True)
    parser.add_argument(
        "--no_task1_identity",
        dest="task1_identity",
        action="store_false",
        help="Also permute task1 (default task1 is identity/original MNIST).",
    )
    return parser.parse_args()


def load_source_config(source_dir: Path) -> dict:
    cfg_path = source_dir / "config.json"
    if cfg_path.exists():
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    return {}


def _extract_client_id(path: Path) -> int:
    nums = re.findall(r"\d+", path.stem)
    if not nums:
        raise ValueError(f"Cannot parse client id from filename: {path.name}")
    return int(nums[-1])


def collect_client_file_pairs(source_dir: Path) -> List[Tuple[int, Path, Path]]:
    train_files = sorted((source_dir / "train").glob("*.npz"))
    test_files = sorted((source_dir / "test").glob("*.npz"))
    if not train_files or not test_files:
        raise FileNotFoundError(f"Missing train/test .npz files under {source_dir}")

    train_map: Dict[int, Path] = {_extract_client_id(p): p for p in train_files}
    test_map: Dict[int, Path] = {_extract_client_id(p): p for p in test_files}
    if len(train_map) != len(train_files) or len(test_map) != len(test_files):
        raise RuntimeError("Duplicate client ids detected while parsing filenames.")

    common_ids = sorted(set(train_map) & set(test_map))
    if not common_ids:
        raise RuntimeError("No overlapping client ids between train and test files.")
    missing_train = sorted(set(test_map) - set(train_map))
    missing_test = sorted(set(train_map) - set(test_map))
    if missing_train or missing_test:
        raise RuntimeError(
            f"Train/test client mismatch. missing_train={missing_train[:5]} missing_test={missing_test[:5]}"
        )
    return [(cid, train_map[cid], test_map[cid]) for cid in common_ids]


def _parse_explicit_seeds(raw: str) -> List[int]:
    raw = (raw or "").strip()
    if not raw:
        return []
    parts = [x.strip() for x in raw.split(",") if x.strip()]
    try:
        return [int(x) for x in parts]
    except ValueError as e:
        raise ValueError("--explicit_task_seeds must be comma-separated integers.") from e


def _permutation_checksum(perm: np.ndarray) -> str:
    return hashlib.sha1(perm.astype(np.int32, copy=False).tobytes()).hexdigest()[:16]


def build_task_specs(num_tasks: int, image_size: int, seed: int, task1_identity: bool, explicit_task_seeds: Sequence[int]):
    if num_tasks < 1:
        raise ValueError("--num_tasks must be >= 1.")
    dim = int(image_size) * int(image_size)
    expected_explicit = num_tasks - 1 if task1_identity else num_tasks
    if explicit_task_seeds and len(explicit_task_seeds) != expected_explicit:
        raise ValueError(
            f"--explicit_task_seeds length mismatch: expected {expected_explicit}, got {len(explicit_task_seeds)}"
        )

    rng = np.random.default_rng(seed)
    seen = set()
    task_specs = []
    explicit_iter = iter(explicit_task_seeds)
    for task_idx in range(1, num_tasks + 1):
        use_identity = bool(task1_identity and task_idx == 1)
        if use_identity:
            perm = np.arange(dim, dtype=np.int64)
            perm_seed = None
        else:
            perm_seed = int(next(explicit_iter)) if explicit_task_seeds else int(rng.integers(0, 2_147_483_647))
            perm = np.random.default_rng(perm_seed).permutation(dim).astype(np.int64)

        checksum = _permutation_checksum(perm)
        if checksum in seen:
            raise RuntimeError(f"Duplicate permutation generated for task {task_idx}.")
        seen.add(checksum)
        task_specs.append(
            {
                "task_idx": int(task_idx),
                "identity": bool(use_identity),
                "permutation_seed": None if perm_seed is None else int(perm_seed),
                "permutation": perm,
                "permutation_checksum": checksum,
                "permutation_preview": perm[:16].astype(int).tolist(),
            }
        )
    return task_specs


def _apply_permutation(x: np.ndarray, perm: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.copy()
    x_np = np.array(x)
    flat = x_np.reshape(x_np.shape[0], -1)
    if flat.shape[1] != perm.shape[0]:
        raise ValueError(
            f"Image flattened dim mismatch: got {flat.shape[1]}, expected {perm.shape[0]} from permutation."
        )
    out = flat[:, perm].reshape(x_np.shape)
    return out.astype(x_np.dtype, copy=False)


def transform_payload(payload: dict, perm: np.ndarray) -> dict:
    x = np.array(payload["x"])
    y = np.array(payload["y"])
    x_out = _apply_permutation(x, perm)
    return {"x": x_out, "y": y.copy()}


def summarize_source_pairs(file_pairs: Sequence[Tuple[int, Path, Path]]) -> Tuple[List[dict], dict]:
    rows: List[dict] = []
    train_counts = []
    test_counts = []
    train_labels_union = set()
    test_labels_union = set()
    sample_shape = None
    x_dtype = None

    for cid, train_path, test_path in file_pairs:
        tr = _load_npz_payload(train_path)
        te = _load_npz_payload(test_path)
        x_tr = np.array(tr["x"])
        y_tr = np.array(tr["y"])
        x_te = np.array(te["x"])
        y_te = np.array(te["y"])
        if sample_shape is None and len(x_tr) > 0:
            sample_shape = list(x_tr[0].shape)
            x_dtype = str(x_tr.dtype)

        tr_n = int(len(y_tr))
        te_n = int(len(y_te))
        train_counts.append(tr_n)
        test_counts.append(te_n)
        train_labels_union.update(np.unique(y_tr).astype(int).tolist())
        test_labels_union.update(np.unique(y_te).astype(int).tolist())
        rows.append(
            {
                "client_id": int(cid),
                "train_samples": tr_n,
                "test_samples": te_n,
                "train_num_labels": int(len(np.unique(y_tr))),
                "test_num_labels": int(len(np.unique(y_te))),
            }
        )

    summary = {
        "num_clients": len(file_pairs),
        "train_total": int(np.sum(train_counts)),
        "test_total": int(np.sum(test_counts)),
        "train_min_client": int(np.min(train_counts)),
        "train_max_client": int(np.max(train_counts)),
        "train_mean_client": float(np.mean(train_counts)),
        "test_min_client": int(np.min(test_counts)),
        "test_max_client": int(np.max(test_counts)),
        "test_mean_client": float(np.mean(test_counts)),
        "labels_present_train": sorted(int(x) for x in train_labels_union),
        "labels_present_test": sorted(int(x) for x in test_labels_union),
        "sample_shape": sample_shape,
        "x_dtype": x_dtype,
    }
    return rows, summary


def write_task_dataset(
    output_dir: Path,
    *,
    file_pairs: Sequence[Tuple[int, Path, Path]],
    source_config: dict,
    task_spec: dict,
    image_size: int,
    num_tasks: int,
    summary_filename: str,
) -> None:
    perm = task_spec["permutation"]
    for _, train_path, test_path in file_pairs:
        tr = _load_npz_payload(train_path)
        te = _load_npz_payload(test_path)
        _save_npz_payload(output_dir / "train" / train_path.name, transform_payload(tr, perm))
        _save_npz_payload(output_dir / "test" / test_path.name, transform_payload(te, perm))

    cfg = copy.deepcopy(source_config)
    cfg["cl_mode"] = "permuted"
    cfg["cl_dataset"] = "MNIST"
    cfg["cl_task_id"] = int(task_spec["task_idx"])
    cfg["cl_num_tasks"] = int(num_tasks)
    cfg["cl_task_identity"] = bool(task_spec["identity"])
    cfg["cl_permutation_seed"] = task_spec["permutation_seed"]
    cfg["cl_permutation_checksum"] = task_spec["permutation_checksum"]
    cfg["cl_permutation_preview"] = [int(v) for v in task_spec["permutation_preview"]]
    cfg["cl_image_size"] = int(image_size)
    cfg["cl_tasks_summary"] = summary_filename
    (output_dir / "config.json").write_text(json.dumps(cfg, ensure_ascii=True, separators=(",", ":")), encoding="utf-8")


def main():
    args = parse_args()
    dataset_root = _resolve_dataset_root(args.dataset_root)
    source_dir = dataset_root / args.source_dataset
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset not found: {source_dir}")

    source_cfg = load_source_config(source_dir)
    if args.require_non_iid and not bool(source_cfg.get("non_iid", False)):
        raise ValueError(f"Source dataset config does not indicate non_iid=true: {source_dir / 'config.json'}")

    file_pairs = collect_client_file_pairs(source_dir)
    explicit_seeds = _parse_explicit_seeds(args.explicit_task_seeds)
    task_specs = build_task_specs(
        num_tasks=args.num_tasks,
        image_size=args.image_size,
        seed=args.seed,
        task1_identity=bool(args.task1_identity),
        explicit_task_seeds=explicit_seeds,
    )
    per_client_stats, source_stats = summarize_source_pairs(file_pairs)

    print(f"Source dataset: {source_dir}")
    print(f"Detected clients: {len(file_pairs)}")
    print(f"Source non_iid flag: {source_cfg.get('non_iid', 'unknown')}")
    print(f"Tasks: {args.num_tasks} (Permuted MNIST), task1_identity={args.task1_identity}")
    print(f"Sample shape: {source_stats.get('sample_shape')} dtype={source_stats.get('x_dtype')}")
    for spec in task_specs:
        seed_txt = "identity" if spec["identity"] else str(spec["permutation_seed"])
        print(
            f"Task {spec['task_idx']}: seed={seed_txt} checksum={spec['permutation_checksum']} "
            f"preview={spec['permutation_preview'][:6]}..."
        )

    summary_path = dataset_root / f"{args.task_prefix.rstrip('-_')}_summary.json"
    summary = {
        "source_dataset": args.source_dataset,
        "source_dir": str(source_dir),
        "num_tasks": int(args.num_tasks),
        "task_prefix": args.task_prefix,
        "seed": int(args.seed),
        "task1_identity": bool(args.task1_identity),
        "preserved_source_non_iid": bool(source_cfg.get("non_iid", False)),
        "image_size": int(args.image_size),
        "source_stats": source_stats,
        "per_client_stats": per_client_stats,
        "task_specs": [
            {
                "task_idx": int(s["task_idx"]),
                "identity": bool(s["identity"]),
                "permutation_seed": s["permutation_seed"],
                "permutation_checksum": s["permutation_checksum"],
                "permutation_preview": [int(v) for v in s["permutation_preview"]],
            }
            for s in task_specs
        ],
    }
    if args.dry_run:
        print(f"[dry-run] Summary would be written to: {summary_path}")
        return

    output_dirs = []
    for spec in task_specs:
        out_name = f"{args.task_prefix}{spec['task_idx']}-npz"
        out_dir = dataset_root / out_name
        if out_dir.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {out_dir} (use --overwrite to allow)")
        output_dirs.append(out_dir)
        print(f"Writing task {spec['task_idx']} -> {out_dir}")
        write_task_dataset(
            out_dir,
            file_pairs=file_pairs,
            source_config=source_cfg,
            task_spec=spec,
            image_size=args.image_size,
            num_tasks=args.num_tasks,
            summary_filename=summary_path.name,
        )

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Done. Summary: {summary_path}")


if __name__ == "__main__":
    main()
