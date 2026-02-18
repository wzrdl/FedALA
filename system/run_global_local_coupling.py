import argparse
import copy
import csv
import json
import math
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import DataLoader

from flcore.servers.serverALA import FedALA
from flcore.trainmodel.models import FedAvgCNN, fastText
from utils.data_utils import read_client_data


VOCAB_SIZE = 98635
HIDDEN_DIM = 32


@dataclass
class TaskOptimum:
    seed: int
    task_idx: int
    dataset: str
    rounds_trained: int
    best_acc: float
    history_acc: List[float]
    state_dict: Dict[str, torch.Tensor]


@dataclass(frozen=True)
class SwitchPair:
    src_idx: int
    dst_idx: int


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_int_csv(raw: str) -> List[int]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one integer.")
    return [int(v) for v in vals]


def parse_str_csv(raw: str) -> List[str]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected at least one dataset name.")
    return vals


def parse_switch_pairs(raw: str, n_tasks: int) -> List[SwitchPair]:
    if not raw.strip():
        return [SwitchPair(i, i + 1) for i in range(1, n_tasks)]

    pairs: List[SwitchPair] = []
    seen = set()
    for token in [t.strip() for t in raw.split(",") if t.strip()]:
        if "-" not in token:
            raise ValueError(f"Invalid switch token '{token}'. Use format like 1-2,1-3")
        a_raw, b_raw = token.split("-", 1)
        a = int(a_raw.strip())
        b = int(b_raw.strip())
        if a == b:
            raise ValueError(f"Invalid switch {token}: source and target must differ.")
        if not (1 <= a <= n_tasks and 1 <= b <= n_tasks):
            raise ValueError(
                f"Invalid switch {token}: task index must be in [1, {n_tasks}]."
            )
        key = (a, b)
        if key not in seen:
            seen.add(key)
            pairs.append(SwitchPair(a, b))

    if not pairs:
        raise ValueError("No valid switch pairs parsed.")
    return pairs


def build_model(model_name: str, dataset_name: str, num_classes: int, device: str):
    lower = dataset_name.lower()
    if model_name == "cnn":
        if "mnist" in lower:
            model = FedAvgCNN(in_features=1, num_classes=num_classes, dim=1024)
        elif "cifar" in lower:
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


def clone_state_dict_to_cpu(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in state_dict.items()}


def build_server_args(
    dataset: str,
    model: torch.nn.Module,
    args: argparse.Namespace,
) -> SimpleNamespace:
    return SimpleNamespace(
        device=args.device,
        dataset=dataset,
        global_rounds=args.max_rounds_per_task,
        model=model,
        num_clients=args.num_clients,
        join_ratio=args.join_ratio,
        random_join_ratio=False,
        eval_gap=1,
        num_classes=args.num_classes,
        batch_size=args.batch_size,
        local_learning_rate=args.local_lr,
        local_steps=args.local_steps,
        eta=args.eta,
        rand_percent=args.rand_percent,
        layer_idx=args.layer_idx,
        ala_threshold=args.ala_threshold,
        ala_num_pre_loss=args.ala_num_pre_loss,
    )


def move_x_to_device(x, device: str):
    if isinstance(x, list):
        x[0] = x[0].to(device)
        return x
    if isinstance(x, tuple):
        return (x[0].to(device), x[1].to(device))
    return x.to(device)


def evaluate_accuracy_on_dataset(
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
            data = read_client_data(dataset, cid, is_train=False)
            loader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)
            for x, y in loader:
                x = move_x_to_device(x, device)
                y = y.to(device)
                logits = model(x)
                total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
                total_count += y.shape[0]

    if total_count == 0:
        raise ValueError(f"No test samples for dataset={dataset}")
    return float(total_correct / total_count)


def evaluate_loss_on_dataset(
    model: torch.nn.Module,
    dataset: str,
    num_clients: int,
    batch_size: int,
    device: str,
    is_train: bool,
) -> float:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        for cid in range(num_clients):
            data = read_client_data(dataset, cid, is_train=is_train)
            loader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)
            for x, y in loader:
                x = move_x_to_device(x, device)
                y = y.to(device)
                logits = model(x)
                total_loss += criterion(logits, y).item()
                total_count += y.shape[0]

    if total_count == 0:
        raise ValueError(f"No samples for dataset={dataset}")
    return float(total_loss / total_count)


def evaluate_accuracy_on_client(
    model: torch.nn.Module,
    dataset: str,
    client_id: int,
    batch_size: int,
    device: str,
    is_train: bool,
) -> Tuple[float, int]:
    model.eval()
    total_correct = 0
    total_count = 0
    with torch.no_grad():
        data = read_client_data(dataset, client_id, is_train=is_train)
        loader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)
        for x, y in loader:
            x = move_x_to_device(x, device)
            y = y.to(device)
            logits = model(x)
            total_correct += (torch.argmax(logits, dim=1) == y).sum().item()
            total_count += y.shape[0]

    if total_count == 0:
        raise ValueError(f"No samples for client={client_id}, dataset={dataset}")
    return float(total_correct / total_count), int(total_count)


def evaluate_loss_on_client(
    model: torch.nn.Module,
    dataset: str,
    client_id: int,
    batch_size: int,
    device: str,
    is_train: bool,
) -> Tuple[float, int]:
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")
    model.eval()
    total_loss = 0.0
    total_count = 0
    with torch.no_grad():
        data = read_client_data(dataset, client_id, is_train=is_train)
        loader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)
        for x, y in loader:
            x = move_x_to_device(x, device)
            y = y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_count += y.shape[0]

    if total_count == 0:
        raise ValueError(f"No samples for client={client_id}, dataset={dataset}")
    return float(total_loss / total_count), int(total_count)


def state_dict_sq_norm_diff(
    state_a: Dict[str, torch.Tensor],
    state_b: Dict[str, torch.Tensor],
) -> float:
    sq = 0.0
    for key, va in state_a.items():
        vb = state_b[key]
        if not torch.is_floating_point(va):
            continue
        diff = (vb.detach().cpu().float() - va.detach().cpu().float()).reshape(-1)
        sq += float(torch.dot(diff, diff).item())
    return sq


def state_dict_delta_vector(
    state_src: Dict[str, torch.Tensor],
    state_dst: Dict[str, torch.Tensor],
) -> np.ndarray:
    chunks = []
    for key, va in state_src.items():
        vb = state_dst[key]
        if not torch.is_floating_point(va):
            continue
        diff = (vb.detach().cpu().float() - va.detach().cpu().float()).reshape(-1)
        chunks.append(diff.numpy())
    if not chunks:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def select_clients_for_source(
    dataset: str,
    num_clients: int,
    k: int,
    seed: int,
) -> List[int]:
    sizes = []
    for cid in range(num_clients):
        n = len(read_client_data(dataset, cid, is_train=True))
        sizes.append((cid, n))

    sizes.sort(key=lambda x: x[1])
    if k >= num_clients:
        return [cid for cid, _ in sizes]

    rng = np.random.default_rng(seed)
    bins = 4
    groups = [[] for _ in range(bins)]
    for idx, (cid, _) in enumerate(sizes):
        b = min(bins - 1, int(idx * bins / num_clients))
        groups[b].append(cid)

    picked: List[int] = []
    per = [k // bins for _ in range(bins)]
    for i in range(k % bins):
        per[i] += 1

    for b in range(bins):
        g = groups[b]
        take = min(per[b], len(g))
        if take > 0:
            picked.extend(rng.choice(g, size=take, replace=False).tolist())

    if len(picked) < k:
        remain = [cid for cid, _ in sizes if cid not in picked]
        need = k - len(picked)
        picked.extend(rng.choice(remain, size=need, replace=False).tolist())

    return sorted(int(x) for x in picked)

def prepare_hessian_library(stable_cl_root: str):
    root = str(Path(stable_cl_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from external_libs.hessian_eigenthings import compute_hessian_eigenthings
    except Exception as exc:
        raise ImportError(
            f"Failed to import hessian_eigenthings from {root}."
        ) from exc
    return compute_hessian_eigenthings


def sample_hessian_data(
    dataset: str,
    client_ids: Sequence[int],
    per_client_samples: int,
    seed: int,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    rng = random.Random(seed)
    sampled: List[Tuple[torch.Tensor, torch.Tensor]] = []
    for cid in client_ids:
        data = read_client_data(dataset, cid, is_train=True)
        if len(data) <= per_client_samples:
            sampled.extend(data)
        else:
            idx = list(range(len(data)))
            rng.shuffle(idx)
            sampled.extend([data[i] for i in idx[:per_client_samples]])

    if not sampled:
        raise ValueError(f"No hessian samples for dataset={dataset}, clients={client_ids}")
    return sampled


def compute_hessian_info(
    model: torch.nn.Module,
    dataset: str,
    client_ids: Sequence[int],
    args: argparse.Namespace,
    seed: int,
    compute_hessian_eigenthings,
) -> Dict[str, object]:
    samples = sample_hessian_data(
        dataset=dataset,
        client_ids=client_ids,
        per_client_samples=args.hessian_subset_per_client,
        seed=seed,
    )
    loader = DataLoader(
        samples,
        batch_size=args.hessian_batch_size,
        drop_last=False,
        shuffle=False,
    )
    criterion = torch.nn.CrossEntropyLoss().to(args.device)
    model_h = copy.deepcopy(model).to(args.device)
    model_h.eval()

    eigvals, eigvecs = compute_hessian_eigenthings(
        model_h,
        loader,
        criterion,
        num_eigenthings=args.hessian_topk,
        full_dataset=args.hessian_full_dataset,
        mode="power_iter",
        use_gpu=(args.device == "cuda"),
        max_samples=args.hessian_max_samples,
        power_iter_steps=args.hessian_power_iter_steps,
        power_iter_err_threshold=args.hessian_power_iter_err_threshold,
        momentum=0.0,
    )
    out = {
        "lambda_top1": float(eigvals[0]),
        "eigvals": np.array(eigvals, dtype=float),
    }
    if args.hessian_topk > 1:
        out["eigvecs"] = np.array(eigvecs, dtype=float)
    return out


def projected_quadratic(
    eigvals: np.ndarray,
    eigvecs: np.ndarray,
    delta_vec: np.ndarray,
) -> float:
    # 0.5 * sum_m lambda_m * (v_m^T delta)^2
    proj = eigvecs @ delta_vec
    val = 0.5 * np.sum(eigvals * (proj ** 2))
    return float(val)


def train_global_optimum(
    init_state: Dict[str, torch.Tensor],
    dataset: str,
    args: argparse.Namespace,
    run_tag: int,
) -> TaskOptimum:
    model = build_model(args.model, dataset, args.num_classes, args.device)
    model.load_state_dict(init_state)
    server_args = build_server_args(dataset=dataset, model=model, args=args)
    server = FedALA(server_args, run_tag)

    best_acc = evaluate_accuracy_on_dataset(
        server.global_model,
        dataset,
        args.num_clients,
        args.batch_size,
        args.device,
    )
    best_state = clone_state_dict_to_cpu(server.global_model.state_dict())
    history = [best_acc]
    bad_rounds = 0
    rounds = 0

    for _ in range(args.max_rounds_per_task):
        rounds += 1
        server.selected_clients = server.select_clients()
        server.send_models()
        for c in server.selected_clients:
            c.train()
        server.receive_models()
        server.aggregate_parameters()

        cur_acc = evaluate_accuracy_on_dataset(
            server.global_model,
            dataset,
            args.num_clients,
            args.batch_size,
            args.device,
        )
        history.append(cur_acc)
        if cur_acc - best_acc > args.global_min_delta:
            best_acc = cur_acc
            best_state = clone_state_dict_to_cpu(server.global_model.state_dict())
            bad_rounds = 0
        else:
            bad_rounds += 1
            if bad_rounds >= args.global_patience:
                break

    return TaskOptimum(
        seed=-1,
        task_idx=-1,
        dataset=dataset,
        rounds_trained=rounds,
        best_acc=float(best_acc),
        history_acc=[float(v) for v in history],
        state_dict=best_state,
    )


def refine_local_optimum(
    init_state: Dict[str, torch.Tensor],
    source_dataset: str,
    client_id: int,
    args: argparse.Namespace,
    seed: int,
) -> Dict[str, object]:
    set_random_seed(seed)
    model = build_model(args.model, source_dataset, args.num_classes, args.device)
    model.load_state_dict(init_state)
    model.train()

    data = read_client_data(source_dataset, client_id, is_train=True)
    loader = DataLoader(data, batch_size=args.batch_size, drop_last=False, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.local_refine_lr)

    best_state = clone_state_dict_to_cpu(model.state_dict())
    best_loss, n_c = evaluate_loss_on_client(
        model, source_dataset, client_id, args.batch_size, args.device, is_train=True
    )
    bad_steps = 0
    used_steps = 0

    for step in range(args.local_refine_max_steps):
        used_steps = step + 1
        for x, y in loader:
            x = move_x_to_device(x, args.device)
            y = y.to(args.device)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

        cur_loss, _ = evaluate_loss_on_client(
            model, source_dataset, client_id, args.batch_size, args.device, is_train=True
        )
        if best_loss - cur_loss > args.local_refine_min_delta:
            best_loss = cur_loss
            best_state = clone_state_dict_to_cpu(model.state_dict())
            bad_steps = 0
        else:
            bad_steps += 1
            if bad_steps >= args.local_refine_patience:
                break

    best_model = build_model(args.model, source_dataset, args.num_classes, args.device)
    best_model.load_state_dict(best_state)
    best_model.eval()

    train_loss, _ = evaluate_loss_on_client(
        best_model, source_dataset, client_id, args.batch_size, args.device, is_train=True
    )
    test_acc, _ = evaluate_accuracy_on_client(
        best_model, source_dataset, client_id, args.batch_size, args.device, is_train=False
    )

    return {
        "state_dict": best_state,
        "train_loss": float(train_loss),
        "test_acc": float(test_acc),
        "n_c": int(n_c),
        "steps": int(used_steps),
    }


def corr_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.shape[0] < 2:
        return 0.0
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.shape[0], dtype=float)
    ranks[order] = np.arange(values.shape[0], dtype=float)
    uniq, inv, counts = np.unique(values, return_inverse=True, return_counts=True)
    for i, c in enumerate(counts):
        if c > 1:
            idx = np.where(inv == i)[0]
            ranks[idx] = float(np.mean(ranks[idx]))
    return ranks + 1.0


def corr_spearman(x: np.ndarray, y: np.ndarray) -> float:
    return corr_pearson(rankdata(x), rankdata(y))


def fit_ols(df: pd.DataFrame, y_col: str, x_cols: Sequence[str]) -> Dict[str, object]:
    y = df[y_col].to_numpy(dtype=float)
    X = df[list(x_cols)].to_numpy(dtype=float)
    X = np.concatenate([np.ones((X.shape[0], 1), dtype=float), X], axis=1)
    names = ["intercept"] + list(x_cols)

    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    n = X.shape[0]
    p = len(x_cols)
    if n > p + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2

    return {
        "n": int(n),
        "p": int(p),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "coef": {k: float(v) for k, v in zip(names, beta)},
    }


def fit_fe_ols(
    df: pd.DataFrame,
    y_col: str,
    x_cols: Sequence[str],
    fe_cols: Sequence[str],
) -> Dict[str, object]:
    y = df[y_col].to_numpy(dtype=float)
    mats = [np.ones((df.shape[0], 1), dtype=float), df[list(x_cols)].to_numpy(dtype=float)]
    names = ["intercept"] + list(x_cols)

    for fe in fe_cols:
        d = pd.get_dummies(df[fe].astype(str), prefix=fe, drop_first=True)
        if d.shape[1] > 0:
            mats.append(d.to_numpy(dtype=float))
            names.extend(d.columns.tolist())

    X = np.concatenate(mats, axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    n = X.shape[0]
    p = len(x_cols)
    if n > p + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2

    x_coef = {k: float(beta[1 + i]) for i, k in enumerate(x_cols)}
    return {
        "n": int(n),
        "p": int(p),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "coef_x": x_coef,
    }


def fit_line(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    if x.shape[0] < 2 or np.std(x) == 0.0:
        return 0.0, float(np.mean(y))
    slope, intercept = np.polyfit(x, y, deg=1)
    return float(slope), float(intercept)


def summarize_by_switch(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    switch_col: str = "switch_label",
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for sw, g in df.groupby(switch_col):
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        if x.shape[0] >= 2 and np.std(x) > 0.0:
            slope, intercept = fit_line(x, y)
            pearson = corr_pearson(x, y)
            spearman = corr_spearman(x, y)
        else:
            slope, intercept = float("nan"), float("nan")
            pearson, spearman = float("nan"), float("nan")
        rows.append(
            {
                "switch_label": str(sw),
                "n": int(len(g)),
                "slope": float(slope),
                "intercept": float(intercept),
                "pearson": float(pearson),
                "spearman": float(spearman),
            }
        )
    return rows


def residualize_within_switch(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    switch_col: str = "switch_label",
) -> pd.DataFrame:
    out = df[[switch_col, x_col, y_col]].copy()
    out["x_res"] = out[x_col] - out.groupby(switch_col)[x_col].transform("mean")
    out["y_res"] = out[y_col] - out.groupby(switch_col)[y_col].transform("mean")
    return out


def fit_switch_interaction_ols(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    switch_col: str = "switch_label",
    control_cols: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    if control_cols is None:
        control_cols = []

    y = df[y_col].to_numpy(dtype=float)
    x = df[x_col].to_numpy(dtype=float)
    sw = df[switch_col].astype(str)
    sw_dummy = pd.get_dummies(sw, prefix="sw", drop_first=True)

    mats = [np.ones((df.shape[0], 1), dtype=float), x.reshape(-1, 1)]
    names = ["intercept", x_col]

    if control_cols:
        mats.append(df[list(control_cols)].to_numpy(dtype=float))
        names.extend(list(control_cols))

    if sw_dummy.shape[1] > 0:
        sw_mat = sw_dummy.to_numpy(dtype=float)
        mats.append(sw_mat)
        names.extend(sw_dummy.columns.tolist())

        inter = sw_mat * x.reshape(-1, 1)
        mats.append(inter)
        names.extend([f"{x_col}:{c}" for c in sw_dummy.columns.tolist()])

    X = np.concatenate(mats, axis=1)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    y_hat = X @ beta
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    n = X.shape[0]
    p = X.shape[1] - 1
    if n > p + 1:
        adj_r2 = float(1.0 - (1.0 - r2) * (n - 1) / (n - p - 1))
    else:
        adj_r2 = r2

    coef = {k: float(v) for k, v in zip(names, beta)}
    base_slope = float(coef.get(x_col, 0.0))

    sw_levels = sorted(sw.unique().tolist())
    ref = sw_levels[0]
    slope_by_switch = {ref: base_slope}
    for s in sw_levels[1:]:
        d_name = f"sw_{s}"
        i_name = f"{x_col}:{d_name}"
        slope_by_switch[s] = float(base_slope + coef.get(i_name, 0.0))

    return {
        "n": int(n),
        "p": int(p),
        "r2": float(r2),
        "adj_r2": float(adj_r2),
        "coef": coef,
        "base_switch": str(ref),
        "slope_by_switch": {k: float(v) for k, v in slope_by_switch.items()},
    }


def plot_switch_decomposition(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    out_path: Path,
    title: str,
    x_label: str,
    y_label: str,
    switch_col: str = "switch_label",
):
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 4.8))

    # Panel 1: switch-wise scatter and switch-wise fit.
    switches = sorted(df[switch_col].astype(str).unique().tolist())
    colors = plt.cm.tab10(np.linspace(0, 1, max(1, len(switches))))
    for idx, sw in enumerate(switches):
        g = df[df[switch_col].astype(str) == sw]
        x = g[x_col].to_numpy(dtype=float)
        y = g[y_col].to_numpy(dtype=float)
        axes[0].scatter(x, y, color=colors[idx], alpha=0.8, s=26, label=sw)
        if len(g) >= 2 and np.std(x) > 0.0:
            a, b = fit_line(x, y)
            xs = np.linspace(float(np.min(x)), float(np.max(x)), 40)
            axes[0].plot(xs, a * xs + b, color=colors[idx], linewidth=1.8)
    axes[0].set_title(f"{title} (Switch-wise)")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel(y_label)
    axes[0].legend(fontsize=8, ncol=2)

    # Panel 2: residual pooled.
    res = residualize_within_switch(df, x_col=x_col, y_col=y_col, switch_col=switch_col)
    xr = res["x_res"].to_numpy(dtype=float)
    yr = res["y_res"].to_numpy(dtype=float)
    axes[1].scatter(xr, yr, alpha=0.8, s=24)
    a, b = fit_line(xr, yr)
    if xr.shape[0] > 0:
        xs = np.linspace(float(np.min(xr)), float(np.max(xr)), 80)
        axes[1].plot(xs, a * xs + b, color="black", linewidth=2)
    axes[1].axhline(0.0, color="gray", linewidth=1)
    axes[1].axvline(0.0, color="gray", linewidth=1)
    axes[1].set_title(f"{title} (Residual-Pooled)")
    axes[1].set_xlabel(f"{x_label} residual")
    axes[1].set_ylabel(f"{y_label} residual")

    # Panel 3: slope by switch.
    sw_rows = summarize_by_switch(df, x_col=x_col, y_col=y_col, switch_col=switch_col)
    sw_df = pd.DataFrame(sw_rows).sort_values("switch_label")
    y_pos = np.arange(sw_df.shape[0])
    axes[2].scatter(sw_df["slope"].to_numpy(dtype=float), y_pos, s=40)
    axes[2].axvline(0.0, color="black", linewidth=1)
    axes[2].set_yticks(y_pos)
    axes[2].set_yticklabels(sw_df["switch_label"].tolist())
    axes[2].set_xlabel("Slope")
    axes[2].set_title(f"{title} (Slope by Switch)")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=230)
    plt.close(fig)


def write_csv(path: Path, rows: List[Dict[str, object]]):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

def plot_experiment_1(global_df: pd.DataFrame, out_dir: Path):
    df = global_df.copy()
    df["log_B_G"] = np.log1p(df["B_G"].astype(float))

    plot_switch_decomposition(
        df=df,
        x_col="log_B_G",
        y_col="F_global_loss",
        out_path=out_dir / "fig_exp1_global_B_relation.png",
        title="Exp1: Global Loss Forgetting vs B_G",
        x_label="log1p(B_G)",
        y_label="F_global_loss",
    )

    plot_switch_decomposition(
        df=df,
        x_col="log_B_G",
        y_col="F_global_acc",
        out_path=out_dir / "fig_exp1_global_B_relation_acc.png",
        title="Exp1: Global Acc Forgetting vs B_G",
        x_label="log1p(B_G)",
        y_label="F_global_acc",
    )


def plot_experiment_2(global_df: pd.DataFrame, out_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))
    x1 = np.log1p(global_df["D_G_sq"].to_numpy(dtype=float))
    x2 = np.log1p(global_df["D_L_mean_sq"].to_numpy(dtype=float))
    y = global_df["F_global_loss"].to_numpy(dtype=float)

    for ax, x, title in [
        (axes[0], x1, "F_global_loss vs D_G"),
        (axes[1], x2, "F_global_loss vs D_L_mean"),
    ]:
        ax.scatter(x, y, c=global_df["seed"].to_numpy(dtype=float), cmap="tab10", alpha=0.85)
        a, b = fit_line(x, y)
        xs = np.linspace(float(np.min(x)), float(np.max(x)), 80)
        ax.plot(xs, a * xs + b, linewidth=2)
        ax.set_xlabel("log1p(shift)")
        ax.set_ylabel("F_global_loss")
        ax.set_title(title)

    # Interaction view
    sc = axes[2].scatter(x1, x2, c=y, cmap="viridis", alpha=0.9)
    cb = fig.colorbar(sc, ax=axes[2])
    cb.set_label("F_global_loss")
    axes[2].set_xlabel("log1p(D_G_sq)")
    axes[2].set_ylabel("log1p(D_L_mean_sq)")
    axes[2].set_title("Global vs Local Shift Coupling")

    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp2_global_shift_relation.png", dpi=230)
    plt.close(fig)


def plot_experiment_3(client_df: pd.DataFrame, out_dir: Path):
    df = client_df.copy()
    df["log_B_G"] = np.log1p(df["B_G"].astype(float))
    plot_switch_decomposition(
        df=df,
        x_col="log_B_G",
        y_col="F_local_loss",
        out_path=out_dir / "fig_exp3_client_forgetting_vs_BG.png",
        title="Exp3: Client Forgetting vs Global B_G",
        x_label="log1p(B_G)",
        y_label="F_local_loss",
    )


def plot_experiment_4(client_df: pd.DataFrame, out_dir: Path):
    df = client_df.copy()
    df["log_B_L"] = np.log1p(df["B_L"].astype(float))
    plot_switch_decomposition(
        df=df,
        x_col="log_B_L",
        y_col="F_local_loss",
        out_path=out_dir / "fig_exp4_client_forgetting_vs_BL.png",
        title="Exp4: Client Forgetting vs Local B_L",
        x_label="log1p(B_L)",
        y_label="F_local_loss",
    )

    # Keep a dedicated coupling map between BG and BL.
    fig = plt.figure(figsize=(6.6, 4.9))
    ax = fig.add_subplot(111)
    x_bg = np.log1p(df["B_G"].to_numpy(dtype=float))
    x_bl = np.log1p(df["B_L"].to_numpy(dtype=float))
    y = df["F_local_loss"].to_numpy(dtype=float)
    hb = ax.hexbin(x_bg, x_bl, C=y, reduce_C_function=np.mean, gridsize=20, cmap="viridis")
    cb = fig.colorbar(hb, ax=ax)
    cb.set_label("mean F_local_loss")
    ax.set_xlabel("log1p(B_G)")
    ax.set_ylabel("log1p(B_L)")
    ax.set_title("Exp4 Coupling Map: B_G and B_L")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp4_BG_BL_coupling.png", dpi=230)
    plt.close(fig)


def plot_experiment_5(model_metrics: Dict[str, Dict[str, object]], out_dir: Path):
    m_names = ["M1_BG", "M2_BL", "M3_BG_BL_inter"]
    adj = [float(model_metrics[k]["adj_r2"]) for k in m_names]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(np.arange(len(m_names)), adj)
    axes[0].set_xticks(np.arange(len(m_names)))
    axes[0].set_xticklabels(m_names, rotation=20)
    axes[0].set_ylabel("Adj R2")
    axes[0].set_title("Experiment 5: Model Comparison")

    coef_names = ["log_B_G", "log_B_L", "interaction"]
    vals = []
    for k in m_names:
        coef = model_metrics[k].get("coef_x", {})
        vals.append([
            float(coef.get("log_B_G", 0.0)),
            float(coef.get("log_B_L", 0.0)),
            float(coef.get("interaction", 0.0)),
        ])
    vals = np.array(vals, dtype=float)

    width = 0.25
    x = np.arange(len(coef_names))
    for i, name in enumerate(m_names):
        axes[1].bar(x + (i - 1) * width, vals[i], width=width, label=name)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(coef_names)
    axes[1].axhline(0.0, color="black", linewidth=1)
    axes[1].set_title("Key Coefficients")
    axes[1].legend(fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "fig_exp5_model_compare.png", dpi=230)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(
        description="Switch-level and client-level forgetting analysis for FedALA."
    )
    parser.add_argument("--task_datasets", type=str, required=True)
    parser.add_argument(
        "--switch_pairs",
        type=str,
        default="",
        help="Comma-separated 1-based pairs like 1-2,1-3. Empty=adjacent only.",
    )
    parser.add_argument("--seeds", type=str, default="0,1")
    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet", "fastText"])
    parser.add_argument("--num_classes", type=int, default=10)

    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--device_id", type=str, default="0")

    parser.add_argument("--num_clients", type=int, default=20)
    parser.add_argument("--join_ratio", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--local_steps", type=int, default=1)
    parser.add_argument("--local_lr", type=float, default=0.1)

    parser.add_argument("--max_rounds_per_task", type=int, default=15)
    parser.add_argument("--global_patience", type=int, default=3)
    parser.add_argument("--global_min_delta", type=float, default=0.002)

    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument("--rand_percent", type=int, default=80)
    parser.add_argument("--layer_idx", type=int, default=1)
    parser.add_argument("--ala_threshold", type=float, default=0.01)
    parser.add_argument("--ala_num_pre_loss", type=int, default=10)

    parser.add_argument("--analyze_clients", type=int, default=8)
    parser.add_argument("--local_refine_lr", type=float, default=0.05)
    parser.add_argument("--local_refine_max_steps", type=int, default=30)
    parser.add_argument("--local_refine_patience", type=int, default=5)
    parser.add_argument("--local_refine_min_delta", type=float, default=1e-3)

    parser.add_argument(
        "--stable_cl_root",
        type=str,
        default=r"C:\Users\24717\Desktop\LLM Forgetting\stable-continual-learning",
    )
    parser.add_argument("--hessian_subset_per_client", type=int, default=64)
    parser.add_argument("--hessian_batch_size", type=int, default=128)
    parser.add_argument("--hessian_max_samples", type=int, default=256)
    parser.add_argument("--hessian_power_iter_steps", type=int, default=8)
    parser.add_argument("--hessian_power_iter_err_threshold", type=float, default=1e-4)
    parser.add_argument("--hessian_topk", type=int, default=1)
    parser.add_argument("--hessian_full_dataset", action="store_true")

    parser.add_argument("--output_dir", type=str, default="./global_local_coupling_logs")
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable; switching to CPU.")
        args.device = "cpu"

    task_datasets = parse_str_csv(args.task_datasets)
    n_tasks = len(task_datasets)
    if n_tasks < 2:
        raise ValueError("Need at least two tasks.")
    switches = parse_switch_pairs(args.switch_pairs, n_tasks)
    seeds = parse_int_csv(args.seeds)

    print("Task datasets:")
    for i, d in enumerate(task_datasets, start=1):
        print(f"  {i}: {d}")
    print("Switch pairs:", [f"{s.src_idx}->{s.dst_idx}" for s in switches])

    compute_hessian_eigenthings = prepare_hessian_library(args.stable_cl_root)

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    global_task_rows: List[Dict[str, object]] = []
    global_switch_rows: List[Dict[str, object]] = []
    client_rows: List[Dict[str, object]] = []
    switch_local_rows: List[Dict[str, object]] = []

    for seed in seeds:
        print(f"\n===== Seed {seed} =====")
        set_random_seed(seed)

        # Same random init for all source-task optima within this seed.
        base_model = build_model(args.model, task_datasets[0], args.num_classes, args.device)
        base_state = clone_state_dict_to_cpu(base_model.state_dict())

        source_indices = sorted({sw.src_idx for sw in switches})

        source_opt_cache: Dict[int, TaskOptimum] = {}
        source_global_hessian_cache: Dict[int, Dict[str, object]] = {}
        source_client_ids_cache: Dict[int, List[int]] = {}
        source_local_opt_cache: Dict[Tuple[int, int], Dict[str, object]] = {}
        source_local_hessian_cache: Dict[Tuple[int, int], Dict[str, object]] = {}

        # 1) Train source-task global optima and cache source Hessian.
        for src_idx in source_indices:
            src_dataset = task_datasets[src_idx - 1]
            print(f"[Seed {seed}] Train source global optimum: task {src_idx} ({src_dataset})")
            opt = train_global_optimum(
                init_state=base_state,
                dataset=src_dataset,
                args=args,
                run_tag=seed * 10000 + src_idx,
            )
            opt.seed = seed
            opt.task_idx = src_idx
            source_opt_cache[src_idx] = opt

            global_task_rows.append(
                {
                    "seed": seed,
                    "task_idx": src_idx,
                    "dataset": src_dataset,
                    "rounds_trained": opt.rounds_trained,
                    "best_acc": opt.best_acc,
                    "history_acc_json": json.dumps(opt.history_acc),
                }
            )

            model_opt = build_model(args.model, src_dataset, args.num_classes, args.device)
            model_opt.load_state_dict(opt.state_dict)
            hess_info = compute_hessian_info(
                model=model_opt,
                dataset=src_dataset,
                client_ids=list(range(args.num_clients)),
                args=args,
                seed=seed * 10000 + src_idx * 17,
                compute_hessian_eigenthings=compute_hessian_eigenthings,
            )
            source_global_hessian_cache[src_idx] = hess_info

            chosen_clients = select_clients_for_source(
                dataset=src_dataset,
                num_clients=args.num_clients,
                k=min(args.analyze_clients, args.num_clients),
                seed=seed * 1000 + src_idx,
            )
            source_client_ids_cache[src_idx] = chosen_clients
            print(
                f"  -> rounds={opt.rounds_trained}, best_acc={opt.best_acc:.4f}, "
                f"lambda_top1={hess_info['lambda_top1']:.4f}, clients={chosen_clients}"
            )

            # 2) Cache local source optimum and local source Hessian for each analyzed client.
            for cid in chosen_clients:
                lopt = refine_local_optimum(
                    init_state=opt.state_dict,
                    source_dataset=src_dataset,
                    client_id=cid,
                    args=args,
                    seed=seed * 100000 + src_idx * 100 + cid,
                )
                source_local_opt_cache[(src_idx, cid)] = lopt

                model_l = build_model(args.model, src_dataset, args.num_classes, args.device)
                model_l.load_state_dict(lopt["state_dict"])
                hess_l = compute_hessian_info(
                    model=model_l,
                    dataset=src_dataset,
                    client_ids=[cid],
                    args=args,
                    seed=seed * 100000 + src_idx * 100 + cid + 50000,
                    compute_hessian_eigenthings=compute_hessian_eigenthings,
                )
                source_local_hessian_cache[(src_idx, cid)] = hess_l

        # 3) For each switch, train switched global optimum and compute global + client metrics.
        switch_global_target_cache: Dict[Tuple[int, int], TaskOptimum] = {}
        for sw in switches:
            src_idx, dst_idx = sw.src_idx, sw.dst_idx
            src_dataset = task_datasets[src_idx - 1]
            dst_dataset = task_datasets[dst_idx - 1]
            switch_label = f"{src_idx}->{dst_idx}"

            src_opt = source_opt_cache[src_idx]

            print(f"[Seed {seed}] Switch {switch_label}: {src_dataset} -> {dst_dataset}")
            if (src_idx, dst_idx) in switch_global_target_cache:
                dst_opt = switch_global_target_cache[(src_idx, dst_idx)]
            else:
                dst_opt = train_global_optimum(
                    init_state=src_opt.state_dict,
                    dataset=dst_dataset,
                    args=args,
                    run_tag=seed * 10000 + src_idx * 100 + dst_idx,
                )
                dst_opt.seed = seed
                dst_opt.task_idx = dst_idx
                switch_global_target_cache[(src_idx, dst_idx)] = dst_opt

            # Global metrics for experiments 1/2.
            model_src = build_model(args.model, src_dataset, args.num_classes, args.device)
            model_src.load_state_dict(src_opt.state_dict)
            model_dst = build_model(args.model, src_dataset, args.num_classes, args.device)
            model_dst.load_state_dict(dst_opt.state_dict)

            f_global_loss = evaluate_loss_on_dataset(
                model=model_dst,
                dataset=src_dataset,
                num_clients=args.num_clients,
                batch_size=args.batch_size,
                device=args.device,
                is_train=True,
            ) - evaluate_loss_on_dataset(
                model=model_src,
                dataset=src_dataset,
                num_clients=args.num_clients,
                batch_size=args.batch_size,
                device=args.device,
                is_train=True,
            )
            f_global_acc = evaluate_accuracy_on_dataset(
                model=model_src,
                dataset=src_dataset,
                num_clients=args.num_clients,
                batch_size=args.batch_size,
                device=args.device,
            ) - evaluate_accuracy_on_dataset(
                model=model_dst,
                dataset=src_dataset,
                num_clients=args.num_clients,
                batch_size=args.batch_size,
                device=args.device,
            )

            d_g_sq = state_dict_sq_norm_diff(src_opt.state_dict, dst_opt.state_dict)
            lambda_s_i = float(source_global_hessian_cache[src_idx]["lambda_top1"])
            b_g = 0.5 * lambda_s_i * d_g_sq

            b_g_topk = None
            if args.hessian_topk > 1 and "eigvecs" in source_global_hessian_cache[src_idx]:
                d_vec = state_dict_delta_vector(src_opt.state_dict, dst_opt.state_dict)
                b_g_topk = projected_quadratic(
                    source_global_hessian_cache[src_idx]["eigvals"],
                    source_global_hessian_cache[src_idx]["eigvecs"],
                    d_vec,
                )

            # Client-level metrics for experiments 3/4/5.
            chosen_clients = source_client_ids_cache[src_idx]
            d_l_values = []
            f_l_values = []
            for cid in chosen_clients:
                src_local = source_local_opt_cache[(src_idx, cid)]
                src_local_h = source_local_hessian_cache[(src_idx, cid)]

                dst_local = refine_local_optimum(
                    init_state=dst_opt.state_dict,
                    source_dataset=src_dataset,
                    client_id=cid,
                    args=args,
                    seed=seed * 1000000 + src_idx * 1000 + dst_idx * 10 + cid,
                )

                f_local_loss = float(dst_local["train_loss"] - src_local["train_loss"])
                f_local_acc = float(src_local["test_acc"] - dst_local["test_acc"])

                d_l_sq = state_dict_sq_norm_diff(src_local["state_dict"], dst_local["state_dict"])
                lambda_l = float(src_local_h["lambda_top1"])
                b_l = 0.5 * lambda_l * d_l_sq

                delta_g_vec = state_dict_delta_vector(src_opt.state_dict, dst_opt.state_dict)
                delta_l_vec = state_dict_delta_vector(src_local["state_dict"], dst_local["state_dict"])
                cos_gl = cosine_similarity_np(delta_g_vec, delta_l_vec)

                client_rows.append(
                    {
                        "seed": seed,
                        "src_idx": src_idx,
                        "dst_idx": dst_idx,
                        "switch_label": switch_label,
                        "client_id": cid,
                        "n_c": src_local["n_c"],
                        "baseline_local_loss": src_local["train_loss"],
                        "baseline_local_acc": src_local["test_acc"],
                        "F_local_loss": f_local_loss,
                        "F_local_acc": f_local_acc,
                        "lambda_L": lambda_l,
                        "D_L_sq": d_l_sq,
                        "B_L": b_l,
                        "B_G": b_g,
                        "D_G_sq": d_g_sq,
                        "lambda_G": lambda_s_i,
                        "cos_G_L": cos_gl,
                    }
                )

                d_l_values.append(d_l_sq)
                f_l_values.append(f_local_loss)

            d_l_arr = np.array(d_l_values, dtype=float)
            f_l_arr = np.array(f_l_values, dtype=float)
            d_l_mean = float(np.mean(d_l_arr))
            d_l_std = float(np.std(d_l_arr))
            d_l_p90 = float(np.percentile(d_l_arr, 90))

            switch_local_rows.append(
                {
                    "seed": seed,
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "switch_label": switch_label,
                    "D_L_mean_sq": d_l_mean,
                    "D_L_std_sq": d_l_std,
                    "D_L_p90_sq": d_l_p90,
                    "F_local_loss_mean": float(np.mean(f_l_arr)),
                    "F_local_loss_std": float(np.std(f_l_arr)),
                    "num_clients": len(chosen_clients),
                }
            )

            global_switch_rows.append(
                {
                    "seed": seed,
                    "src_idx": src_idx,
                    "dst_idx": dst_idx,
                    "switch_label": switch_label,
                    "src_dataset": src_dataset,
                    "dst_dataset": dst_dataset,
                    "lambda_G": lambda_s_i,
                    "D_G_sq": d_g_sq,
                    "B_G": b_g,
                    "B_G_topk": b_g_topk if b_g_topk is not None else "",
                    "F_global_loss": f_global_loss,
                    "F_global_acc": f_global_acc,
                    "D_L_mean_sq": d_l_mean,
                    "D_L_std_sq": d_l_std,
                    "D_L_p90_sq": d_l_p90,
                }
            )

    # Build dataframes for analysis.
    global_df = pd.DataFrame(global_switch_rows)
    client_df = pd.DataFrame(client_rows)

    if global_df.empty or client_df.empty:
        raise RuntimeError("No results generated. Check task/switch settings.")

    # -------- Experiment 1: B_G vs global forgetting (switch-aware) --------
    g1 = global_df.copy()
    g1["log_B_G"] = np.log1p(g1["B_G"].astype(float))
    res_loss = residualize_within_switch(g1, x_col="log_B_G", y_col="F_global_loss")
    res_acc = residualize_within_switch(g1, x_col="log_B_G", y_col="F_global_acc")

    exp1 = {
        "pooled_corr_Floss_BG": {
            "pearson": corr_pearson(g1["B_G"].to_numpy(float), g1["F_global_loss"].to_numpy(float)),
            "spearman": corr_spearman(g1["B_G"].to_numpy(float), g1["F_global_loss"].to_numpy(float)),
        },
        "pooled_corr_Facc_BG": {
            "pearson": corr_pearson(g1["B_G"].to_numpy(float), g1["F_global_acc"].to_numpy(float)),
            "spearman": corr_spearman(g1["B_G"].to_numpy(float), g1["F_global_acc"].to_numpy(float)),
        },
        "switch_summary_loss": summarize_by_switch(
            g1, x_col="log_B_G", y_col="F_global_loss", switch_col="switch_label"
        ),
        "switch_summary_acc": summarize_by_switch(
            g1, x_col="log_B_G", y_col="F_global_acc", switch_col="switch_label"
        ),
        "residual_corr_loss": {
            "pearson": corr_pearson(res_loss["x_res"].to_numpy(float), res_loss["y_res"].to_numpy(float)),
            "spearman": corr_spearman(res_loss["x_res"].to_numpy(float), res_loss["y_res"].to_numpy(float)),
        },
        "residual_corr_acc": {
            "pearson": corr_pearson(res_acc["x_res"].to_numpy(float), res_acc["y_res"].to_numpy(float)),
            "spearman": corr_spearman(res_acc["x_res"].to_numpy(float), res_acc["y_res"].to_numpy(float)),
        },
        "ols_Floss_on_BG": fit_ols(g1, "F_global_loss", ["log_B_G"]),
        "ols_Facc_on_BG": fit_ols(g1, "F_global_acc", ["log_B_G"]),
        "ols_Floss_switch_interaction": fit_switch_interaction_ols(
            g1, y_col="F_global_loss", x_col="log_B_G", switch_col="switch_label"
        ),
        "ols_Facc_switch_interaction": fit_switch_interaction_ols(
            g1, y_col="F_global_acc", x_col="log_B_G", switch_col="switch_label"
        ),
    }

    # -------- Experiment 2: global forgetting vs global/local shift --------
    g2 = global_df.copy()
    g2["log_D_G"] = np.log1p(g2["D_G_sq"].astype(float))
    g2["log_D_L_mean"] = np.log1p(g2["D_L_mean_sq"].astype(float))
    g2["interaction"] = g2["log_D_G"] * g2["log_D_L_mean"]

    exp2 = {
        "ols_Floss_on_DG": fit_ols(g2, "F_global_loss", ["log_D_G"]),
        "ols_Floss_on_DLmean": fit_ols(g2, "F_global_loss", ["log_D_L_mean"]),
        "ols_Floss_on_DG_DL_inter": fit_ols(
            g2,
            "F_global_loss",
            ["log_D_G", "log_D_L_mean", "interaction"],
        ),
    }

    # -------- Experiment 3/4/5: client-level --------
    c = client_df.copy()
    c["log_B_G"] = np.log1p(c["B_G"].astype(float))
    c["log_B_L"] = np.log1p(c["B_L"].astype(float))
    c["log_n_c"] = np.log(np.maximum(c["n_c"].astype(float), 1.0))
    c["interaction"] = c["log_B_G"] * c["log_B_L"]

    # Experiment 3: F_local ~ B_G + controls + FE(seed,switch,client)
    model_exp3 = fit_fe_ols(
        c,
        y_col="F_local_loss",
        x_cols=["log_B_G", "log_n_c", "baseline_local_loss"],
        fe_cols=["seed", "switch_label", "client_id"],
    )

    # Experiment 4: F_local ~ B_L + controls + FE
    model_exp4 = fit_fe_ols(
        c,
        y_col="F_local_loss",
        x_cols=["log_B_L", "log_n_c", "baseline_local_loss"],
        fe_cols=["seed", "switch_label", "client_id"],
    )

    # Experiment 5: compare BG-only, BL-only, BG+BL+interaction
    model_m1 = fit_fe_ols(
        c,
        y_col="F_local_loss",
        x_cols=["log_B_G", "log_n_c", "baseline_local_loss"],
        fe_cols=["seed", "switch_label", "client_id"],
    )
    model_m2 = fit_fe_ols(
        c,
        y_col="F_local_loss",
        x_cols=["log_B_L", "log_n_c", "baseline_local_loss"],
        fe_cols=["seed", "switch_label", "client_id"],
    )
    model_m3 = fit_fe_ols(
        c,
        y_col="F_local_loss",
        x_cols=["log_B_G", "log_B_L", "interaction", "log_n_c", "baseline_local_loss"],
        fe_cols=["seed", "switch_label", "client_id"],
    )

    res_bg = residualize_within_switch(c, x_col="log_B_G", y_col="F_local_loss")
    res_bl = residualize_within_switch(c, x_col="log_B_L", y_col="F_local_loss")

    exp3 = {
        "pooled_corr": {
            "pearson": corr_pearson(c["B_G"].to_numpy(float), c["F_local_loss"].to_numpy(float)),
            "spearman": corr_spearman(c["B_G"].to_numpy(float), c["F_local_loss"].to_numpy(float)),
        },
        "switch_summary": summarize_by_switch(
            c, x_col="log_B_G", y_col="F_local_loss", switch_col="switch_label"
        ),
        "residual_corr": {
            "pearson": corr_pearson(res_bg["x_res"].to_numpy(float), res_bg["y_res"].to_numpy(float)),
            "spearman": corr_spearman(res_bg["x_res"].to_numpy(float), res_bg["y_res"].to_numpy(float)),
        },
        "switch_interaction_model": fit_switch_interaction_ols(
            c,
            y_col="F_local_loss",
            x_col="log_B_G",
            switch_col="switch_label",
            control_cols=["log_n_c", "baseline_local_loss"],
        ),
        "fe_model": model_exp3,
    }

    exp4 = {
        "pooled_corr": {
            "pearson": corr_pearson(c["B_L"].to_numpy(float), c["F_local_loss"].to_numpy(float)),
            "spearman": corr_spearman(c["B_L"].to_numpy(float), c["F_local_loss"].to_numpy(float)),
        },
        "switch_summary": summarize_by_switch(
            c, x_col="log_B_L", y_col="F_local_loss", switch_col="switch_label"
        ),
        "residual_corr": {
            "pearson": corr_pearson(res_bl["x_res"].to_numpy(float), res_bl["y_res"].to_numpy(float)),
            "spearman": corr_spearman(res_bl["x_res"].to_numpy(float), res_bl["y_res"].to_numpy(float)),
        },
        "switch_interaction_model": fit_switch_interaction_ols(
            c,
            y_col="F_local_loss",
            x_col="log_B_L",
            switch_col="switch_label",
            control_cols=["log_n_c", "baseline_local_loss"],
        ),
        "fe_model": model_exp4,
    }
    exp5 = {
        "M1_BG": model_m1,
        "M2_BL": model_m2,
        "M3_BG_BL_inter": model_m3,
    }

    # -------- Experiment 6 (optional robustness) --------
    exp6 = {}
    if args.hessian_topk > 1 and "B_G_topk" in global_df.columns:
        valid = global_df[global_df["B_G_topk"] != ""].copy()
        if not valid.empty:
            valid["B_G_topk"] = valid["B_G_topk"].astype(float)
            exp6 = {
                "corr_Floss_BG_topk": {
                    "pearson": corr_pearson(valid["B_G_topk"].to_numpy(float), valid["F_global_loss"].to_numpy(float)),
                    "spearman": corr_spearman(valid["B_G_topk"].to_numpy(float), valid["F_global_loss"].to_numpy(float)),
                },
                "ols_Floss_on_BG_topk": fit_ols(
                    valid.assign(log_B_G_topk=np.log1p(valid["B_G_topk"])),
                    "F_global_loss",
                    ["log_B_G_topk"],
                ),
            }

    # Save tables.
    write_csv(output_dir / "global_task_optima.csv", global_task_rows)
    write_csv(output_dir / "global_switch_metrics.csv", global_switch_rows)
    write_csv(output_dir / "client_switch_metrics.csv", client_rows)
    write_csv(output_dir / "switch_local_aggregates.csv", switch_local_rows)

    # Save analysis summary.
    summary = {
        "config": {
            "task_datasets": task_datasets,
            "switch_pairs": [f"{s.src_idx}->{s.dst_idx}" for s in switches],
            "seeds": seeds,
            "num_clients": args.num_clients,
            "analyze_clients": args.analyze_clients,
        },
        "experiment_1": exp1,
        "experiment_2": exp2,
        "experiment_3": exp3,
        "experiment_4": exp4,
        "experiment_5": exp5,
        "experiment_6": exp6,
    }
    with open(output_dir / "analysis_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=True, indent=2)

    # Draw figures.
    plot_experiment_1(global_df, output_dir)
    plot_experiment_2(global_df, output_dir)
    plot_experiment_3(client_df, output_dir)
    plot_experiment_4(client_df, output_dir)
    plot_experiment_5(exp5, output_dir)

    print("\n===== Completed =====")
    print(f"Global switches: {len(global_df)}")
    print(f"Client rows: {len(client_df)}")
    print(f"Output dir: {output_dir}")
    print(
        "Exp1 corr(F_global_loss, B_G): "
        f"pearson={exp1['pooled_corr_Floss_BG']['pearson']:.4f}, "
        f"spearman={exp1['pooled_corr_Floss_BG']['spearman']:.4f}"
    )
    print(
        "Exp3 switch-interaction base slope(log_B_G): "
        f"{exp3['switch_interaction_model']['coef'].get('log_B_G', 0.0):.4f}, "
        f"adj_R2={exp3['switch_interaction_model']['adj_r2']:.4f}"
    )
    print(
        "Exp4 switch-interaction base slope(log_B_L): "
        f"{exp4['switch_interaction_model']['coef'].get('log_B_L', 0.0):.4f}, "
        f"adj_R2={exp4['switch_interaction_model']['adj_r2']:.4f}"
    )
    print(
        "Exp5 model adj_R2: "
        f"M1={exp5['M1_BG']['adj_r2']:.4f}, "
        f"M2={exp5['M2_BL']['adj_r2']:.4f}, "
        f"M3={exp5['M3_BG_BL_inter']['adj_r2']:.4f}"
    )


if __name__ == "__main__":
    main()
# python -u system/run_global_local_coupling.py --task_datasets permuted-mnist-task1-npz,permuted-mnist-task2-npz,permuted-mnist-task3-npz,permuted-mnist-task4-npz --switch_pairs 1-2,2-3,3-4 --seeds 0,1,2 --model cnn --num_classes 10 --device cuda --device_id 0 --num_clients 8 --join_ratio 1.0 --batch_size 10 --local_steps 1 --local_lr 0.1 --max_rounds_per_task 60 --global_patience 3 --global_min_delta 0.002 --analyze_clients 8 --local_refine_lr 0.05 --local_refine_max_steps 20 --local_refine_patience 4 --hessian_subset_per_client 64 --hessian_batch_size 64 --hessian_max_samples 256 --hessian_power_iter_steps 6 --stable_cl_root "C:\Users\24717\Desktop\LLM Forgetting\stable-continual-learning" --output_dir ./system/global_local_coupling_logs_v2