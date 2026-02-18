import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from torchvision.transforms.functional import InterpolationMode, rotate


def _load_npz_data(npz_file: Path):
    with np.load(npz_file, allow_pickle=True) as data:
        payload = data["data"].tolist()
    return payload


def _save_npz_data(npz_file: Path, payload: dict):
    npz_file.parent.mkdir(parents=True, exist_ok=True)
    with open(npz_file, "wb") as f:
        np.savez_compressed(f, data=payload)


def _apply_permutation(x: np.ndarray, permutation: np.ndarray) -> np.ndarray:
    flat = x.reshape(x.shape[0], -1)
    out = flat[:, permutation].reshape(x.shape)
    return out.astype(np.float32)


def _apply_rotation(x: np.ndarray, angle: float) -> np.ndarray:
    x_tensor = torch.from_numpy(x).float()
    out = torch.empty_like(x_tensor)

    # Rotate each sample to keep the transform deterministic and explicit.
    for idx in range(x_tensor.shape[0]):
        out[idx] = rotate(
            x_tensor[idx],
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0.0,
        )
    return out.numpy().astype(np.float32)


def _transform_payload(payload: dict, mode: str, task_spec: dict) -> dict:
    x = np.array(payload["x"], dtype=np.float32)
    y = np.array(payload["y"], dtype=np.int64)

    if mode == "permuted":
        x_t = _apply_permutation(x, task_spec["permutation"])
    elif mode == "rotated":
        x_t = _apply_rotation(x, task_spec["angle"])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return {"x": x_t, "y": y}


def _prepare_task_specs(args):
    rng = np.random.default_rng(args.seed)
    task_specs = []

    if args.mode == "permuted":
        task1_seed = args.task1_seed if args.task1_seed is not None else int(rng.integers(0, 10_000_000))
        task2_seed = args.task2_seed if args.task2_seed is not None else int(rng.integers(0, 10_000_000))
        p1 = np.random.default_rng(task1_seed).permutation(args.image_size * args.image_size)
        p2 = np.random.default_rng(task2_seed).permutation(args.image_size * args.image_size)
        task_specs.append({"task_id": 1, "permutation_seed": task1_seed, "permutation": p1})
        task_specs.append({"task_id": 2, "permutation_seed": task2_seed, "permutation": p2})
    else:
        if args.task1_angle is not None:
            angle1 = float(args.task1_angle)
        else:
            angle1 = float(rng.uniform(0.0, 180.0))
        if args.task2_angle is not None:
            angle2 = float(args.task2_angle)
        else:
            angle2 = float(rng.uniform(0.0, 180.0))
        task_specs.append({"task_id": 1, "angle": angle1})
        task_specs.append({"task_id": 2, "angle": angle2})

    return task_specs


def _default_output_names(mode: str):
    if mode == "permuted":
        return "mnist-permuted-task1-npz", "mnist-permuted-task2-npz"
    return "mnist-rotated-task1-npz", "mnist-rotated-task2-npz"


def main():
    parser = argparse.ArgumentParser(
        description="Create 2-task continual-learning datasets from an existing MNIST federated dataset."
    )
    parser.add_argument("--dataset_root", type=str, default="../dataset")
    parser.add_argument("--source_dataset", type=str, default="mnist-0.1-npz")
    parser.add_argument("--mode", type=str, default="permuted", choices=["permuted", "rotated"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--image_size", type=int, default=28)

    # Optional explicit task specs.
    parser.add_argument("--task1_seed", type=int, default=None, help="Permutation seed for task 1.")
    parser.add_argument("--task2_seed", type=int, default=None, help="Permutation seed for task 2.")
    parser.add_argument("--task1_angle", type=float, default=None, help="Rotation angle for task 1.")
    parser.add_argument("--task2_angle", type=float, default=None, help="Rotation angle for task 2.")

    parser.add_argument("--task1_name", type=str, default=None)
    parser.add_argument("--task2_name", type=str, default=None)
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root).resolve()
    source_dir = dataset_root / args.source_dataset
    if not source_dir.exists():
        raise FileNotFoundError(f"Source dataset does not exist: {source_dir}")

    default_task1_name, default_task2_name = _default_output_names(args.mode)
    task1_name = args.task1_name or default_task1_name
    task2_name = args.task2_name or default_task2_name

    task_specs = _prepare_task_specs(args)
    output_dirs = [dataset_root / task1_name, dataset_root / task2_name]

    print("Source dataset:", source_dir)
    print("Task 1 output:", output_dirs[0])
    print("Task 2 output:", output_dirs[1])
    if args.mode == "permuted":
        print("Task 1 permutation seed:", task_specs[0]["permutation_seed"])
        print("Task 2 permutation seed:", task_specs[1]["permutation_seed"])
    else:
        print("Task 1 rotation angle:", task_specs[0]["angle"])
        print("Task 2 rotation angle:", task_specs[1]["angle"])

    train_files = sorted((source_dir / "train").glob("*.npz"))
    test_files = sorted((source_dir / "test").glob("*.npz"))
    if not train_files or not test_files:
        raise FileNotFoundError(
            f"Source dataset must contain train/test npz files. Missing under: {source_dir}"
        )

    for task_spec, out_dir in zip(task_specs, output_dirs):
        (out_dir / "train").mkdir(parents=True, exist_ok=True)
        (out_dir / "test").mkdir(parents=True, exist_ok=True)

        for src_file in train_files:
            payload = _load_npz_data(src_file)
            transformed = _transform_payload(payload, args.mode, task_spec)
            _save_npz_data(out_dir / "train" / src_file.name, transformed)

        for src_file in test_files:
            payload = _load_npz_data(src_file)
            transformed = _transform_payload(payload, args.mode, task_spec)
            _save_npz_data(out_dir / "test" / src_file.name, transformed)

        src_cfg = source_dir / "config.json"
        if src_cfg.exists():
            cfg = json.loads(src_cfg.read_text(encoding="utf-8"))
        else:
            cfg = {}
        cfg = copy.deepcopy(cfg)
        cfg["cl_mode"] = args.mode
        cfg["cl_task_id"] = task_spec["task_id"]
        if args.mode == "permuted":
            cfg["cl_permutation_seed"] = task_spec["permutation_seed"]
        else:
            cfg["cl_rotation_angle"] = task_spec["angle"]
        (out_dir / "config.json").write_text(
            json.dumps(cfg, ensure_ascii=True, separators=(",", ":")),
            encoding="utf-8",
        )

    print("Done.")


if __name__ == "__main__":
    main()
