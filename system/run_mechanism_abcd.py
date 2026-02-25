import argparse
import copy
import csv
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader

from flcore.clients.clientALA import clientALA
from flcore.trainmodel.models import FedAvgCNN, fastText
from utils.data_utils import read_client_data


VOCAB_SIZE = 98635
HIDDEN_DIM = 32


@dataclass(frozen=True)
class ExperimentVariant:
    exp_id: str
    run_name: str
    personalization_enabled: bool
    freeze_mask: str
    probe_enabled: bool


EXPERIMENT_VARIANTS: Dict[str, ExperimentVariant] = {
    "A": ExperimentVariant("A", "A_Full_FedALA", True, "none", True),
    "B": ExperimentVariant("B", "B_No_Pers", False, "none", True),
    "C": ExperimentVariant("C", "C_Freeze_P", True, "freeze_P", False),
    "D": ExperimentVariant("D", "D_Freeze_S", True, "freeze_S", False),
}


def set_random_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_csv(raw: str) -> List[str]:
    vals = [v.strip() for v in raw.split(",") if v.strip()]
    if not vals:
        raise ValueError("Expected a non-empty comma-separated list.")
    return vals


def parse_experiments(raw: str) -> List[ExperimentVariant]:
    out: List[ExperimentVariant] = []
    seen = set()
    for token in parse_csv(raw):
        exp_id = token.upper()
        if exp_id not in EXPERIMENT_VARIANTS:
            raise ValueError(f"Unsupported experiment id '{token}'. Use A/B/C/D.")
        if exp_id in seen:
            continue
        seen.add(exp_id)
        out.append(EXPERIMENT_VARIANTS[exp_id])
    return out


def move_x_to_device(x, device: str):
    if isinstance(x, list):
        x[0] = x[0].to(device)
        return x
    if isinstance(x, tuple):
        return (x[0].to(device), x[1].to(device))
    return x.to(device)


def apply_bn_policy(model: torch.nn.Module, bn_policy: str):
    if bn_policy == "freeze_stats":
        for m in model.modules():
            if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
                m.eval()
    elif bn_policy in ("none", "default"):
        return
    else:
        raise ValueError(f"Unsupported bn_policy: {bn_policy}")


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


def evaluate_accuracy_on_client(
    model: torch.nn.Module,
    dataset: str,
    client_id: int,
    batch_size: int,
    device: str,
    is_train: bool = False,
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


def snapshot_params_cpu(model: torch.nn.Module) -> List[torch.Tensor]:
    return [p.detach().cpu().float().clone() for p in model.parameters()]


def param_l2_distance(model: torch.nn.Module, param_snapshot: Sequence[torch.Tensor]) -> float:
    sq = 0.0
    for p, p0 in zip(model.parameters(), param_snapshot):
        if not torch.is_floating_point(p):
            continue
        d = (p.detach().cpu().float() - p0.detach().cpu().float()).reshape(-1)
        sq += float(torch.dot(d, d).item())
    return float(math.sqrt(max(sq, 0.0)))


def get_fixed_batch_for_dataset_client(
    dataset: str,
    client_id: int,
    batch_size: int,
    seed: int,
    is_train: bool = True,
) -> Tuple[Optional[Tuple[object, torch.Tensor]], int]:
    data = read_client_data(dataset, client_id, is_train=is_train)
    if len(data) == 0:
        return None, 0
    loader = DataLoader(data, batch_size=batch_size, drop_last=False, shuffle=False)
    batches = [clone_batch_cpu(b) for b in loader]
    if not batches:
        return None, 0
    offset = int(seed % len(batches))
    return clone_batch_cpu(batches[offset]), offset


def infer_p_flags(num_params: int, layer_idx: int) -> List[bool]:
    if layer_idx < 0:
        raise ValueError("layer_idx must be >= 0.")
    if num_params <= 0:
        return []
    if layer_idx == 0:
        # Match utils/ALA.py semantics (layer_idx=0 => all params in P).
        return [True] * num_params
    layer_idx = min(layer_idx, num_params)
    split = num_params - layer_idx
    return [i >= split for i in range(num_params)]


def compute_update_norms(
    model: torch.nn.Module,
    global_snapshot: Sequence[torch.Tensor],
    p_flags: Sequence[bool],
) -> Dict[str, float]:
    total_sq = 0.0
    p_sq = 0.0
    s_sq = 0.0
    for idx, (param, p0) in enumerate(zip(model.parameters(), global_snapshot)):
        if not torch.is_floating_point(param):
            continue
        diff = (param.detach().cpu().float() - p0).reshape(-1)
        sq = float(torch.dot(diff, diff).item())
        total_sq += sq
        if idx < len(p_flags) and p_flags[idx]:
            p_sq += sq
        else:
            s_sq += sq
    total = math.sqrt(max(total_sq, 0.0))
    p_norm = math.sqrt(max(p_sq, 0.0))
    s_norm = math.sqrt(max(s_sq, 0.0))
    return {
        "U_norm": float(total),
        "U_P_norm": float(p_norm),
        "U_S_norm": float(s_norm),
        "U_P_ratio": float(p_norm / (total + 1e-12)),
    }


def mean_or_nan(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def std_or_nan(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else float("nan")


def weighted_mean_or_nan(values: Sequence[float], weights: Sequence[float]) -> float:
    if not values or not weights or len(values) != len(weights):
        return float("nan")
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w) & (w > 0)
    if not np.any(mask):
        return float("nan")
    return float(np.average(v[mask], weights=w[mask]))


def write_csv(path: Path, rows: List[Dict[str, object]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    seen = set()
    for row in rows:
        for k in row.keys():
            if k not in seen:
                seen.add(k)
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _clone_x_cpu(x):
    if torch.is_tensor(x):
        return x.detach().cpu().clone()
    if isinstance(x, list):
        return [_clone_x_cpu(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_clone_x_cpu(v) for v in x)
    return copy.deepcopy(x)


def clone_batch_cpu(batch):
    x, y = batch
    x_c = _clone_x_cpu(x)
    y_c = y.detach().cpu().clone() if torch.is_tensor(y) else copy.deepcopy(y)
    return x_c, y_c


def flatten_float_tensors_cpu(tensors: Sequence[torch.Tensor]) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for t in tensors:
        if not torch.is_tensor(t) or not torch.is_floating_point(t):
            continue
        chunks.append(t.detach().cpu().float().reshape(-1).numpy())
    if not chunks:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def flatten_model_update_cpu(
    model: torch.nn.Module,
    start_snapshot: Sequence[torch.Tensor],
    mask_flags: Optional[Sequence[bool]] = None,
) -> np.ndarray:
    chunks: List[np.ndarray] = []
    for i, (p, p0) in enumerate(zip(model.parameters(), start_snapshot)):
        if not torch.is_floating_point(p):
            continue
        diff = (p.detach().cpu().float() - p0.detach().cpu().float()).reshape(-1)
        if mask_flags is not None and i < len(mask_flags) and not mask_flags[i]:
            diff = torch.zeros_like(diff)
        chunks.append(diff.numpy())
    if not chunks:
        return np.zeros((1,), dtype=np.float32)
    return np.concatenate(chunks, axis=0)


def cosine_similarity_np(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size:
        n = min(a.size, b.size)
        a = a[:n]
        b = b[:n]
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_mean_loss_on_batches(
    model: torch.nn.Module,
    batches: Sequence[Tuple[object, torch.Tensor]],
    loss_fn: torch.nn.Module,
    device: str,
) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for batch in batches:
            x, y = clone_batch_cpu(batch)
            x = move_x_to_device(x, device)
            y = y.to(device)
            losses.append(float(loss_fn(model(x), y).item()))
    return float(np.mean(losses)) if losses else float("nan")


def get_probe_batches_from_client(
    client: "MechanismClientALA",
    steps: int,
    batch_size: int,
    seed: int,
    fixed_batch: bool,
) -> Tuple[List[Tuple[object, torch.Tensor]], int]:
    loader = client.load_train_data(batch_size=batch_size)
    all_batches = [clone_batch_cpu(b) for b in loader]
    if not all_batches:
        return [], 0

    if fixed_batch:
        offset = seed % len(all_batches)
    else:
        rng = random.Random(seed)
        offset = rng.randrange(len(all_batches))

    probe_batches: List[Tuple[object, torch.Tensor]] = []
    for i in range(max(1, steps)):
        probe_batches.append(clone_batch_cpu(all_batches[(offset + i) % len(all_batches)]))
    return probe_batches, int(offset)


def _float_param_list(model: torch.nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if torch.is_floating_point(p)]


def _flatten_grads(grads: Sequence[Optional[torch.Tensor]], params: Sequence[torch.nn.Parameter]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for g, p in zip(grads, params):
        if not torch.is_floating_point(p):
            continue
        if g is None:
            chunks.append(torch.zeros_like(p).reshape(-1))
        else:
            chunks.append(g.reshape(-1))
    if not chunks:
        return torch.zeros(1, device=params[0].device if params else "cpu")
    return torch.cat(chunks)


def _hessian_top_eig_first_batch(
    model: torch.nn.Module,
    batch: Tuple[object, torch.Tensor],
    loss_fn: torch.nn.Module,
    device: str,
    power_steps: int,
    bn_policy: str = "freeze_stats",
) -> Tuple[float, Optional[np.ndarray]]:
    params = _float_param_list(model)
    if not params:
        return float("nan"), None

    model.train()
    apply_bn_policy(model, bn_policy)
    x, y = clone_batch_cpu(batch)
    x = move_x_to_device(x, device)
    y = y.to(device)
    loss = loss_fn(model(x), y)
    grads = torch.autograd.grad(loss, params, create_graph=True, allow_unused=True)
    g_flat = _flatten_grads(grads, params)
    dim = g_flat.numel()
    if dim == 0:
        return float("nan"), None

    v = torch.randn(dim, device=g_flat.device)
    v = v / (v.norm() + 1e-12)
    eigval = torch.tensor(0.0, device=g_flat.device)

    for _ in range(max(1, power_steps)):
        gv = torch.dot(g_flat, v)
        hvp = torch.autograd.grad(gv, params, retain_graph=True, allow_unused=True)
        h_flat = _flatten_grads(hvp, params)
        nrm = h_flat.norm()
        if float(nrm.item()) == 0.0:
            return 0.0, None
        v = h_flat / (nrm + 1e-12)
        eigval = torch.dot(v, h_flat)

    return float(eigval.detach().cpu().item()), v.detach().cpu().float().numpy()


def run_short_sgd_probe(
    start_model: torch.nn.Module,
    batches: Sequence[Tuple[object, torch.Tensor]],
    *,
    device: str,
    loss_fn: torch.nn.Module,
    lr: float,
    momentum: float,
    weight_decay: float,
    grad_clip_norm: Optional[float],
    frozen_flags: Sequence[bool],
    p_flags: Sequence[bool],
    top_eig_power_steps: int,
    top_eig_enabled: bool,
    bn_policy: str,
) -> Dict[str, object]:
    model = copy.deepcopy(start_model).to(device)
    model.train()
    apply_bn_policy(model, bn_policy)
    params = list(model.parameters())
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    start_snapshot = snapshot_params_cpu(model)

    first_grad_vec: Optional[np.ndarray] = None
    losses_during: List[float] = []

    top_eigval = float("nan")
    top_eig_proj_abs = float("nan")
    top_eig_proj_ratio = float("nan")

    if top_eig_enabled and batches:
        top_eigval, top_vec = _hessian_top_eig_first_batch(
            model=model,
            batch=batches[0],
            loss_fn=loss_fn,
            device=device,
            power_steps=top_eig_power_steps,
            bn_policy=bn_policy,
        )
    else:
        top_vec = None

    loss_before = evaluate_mean_loss_on_batches(model, batches, loss_fn, device)

    for step_idx, batch in enumerate(batches):
        x, y = clone_batch_cpu(batch)
        x = move_x_to_device(x, device)
        y = y.to(device)

        optimizer.zero_grad()
        out = model(x)
        loss = loss_fn(out, y)
        losses_during.append(float(loss.item()))
        loss.backward()

        for p, frozen in zip(params, frozen_flags):
            if frozen and p.grad is not None:
                p.grad.zero_()

        if step_idx == 0:
            grad_chunks: List[np.ndarray] = []
            for p in params:
                if not torch.is_floating_point(p):
                    continue
                g = p.grad
                if g is None:
                    grad_chunks.append(np.zeros(int(p.numel()), dtype=np.float32))
                else:
                    grad_chunks.append(g.detach().cpu().float().reshape(-1).numpy())
            if grad_chunks:
                first_grad_vec = np.concatenate(grad_chunks, axis=0)

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(params, grad_clip_norm)

        optimizer.step()

    loss_after = evaluate_mean_loss_on_batches(model, batches, loss_fn, device)
    norm_stats = compute_update_norms(model, start_snapshot, p_flags)
    update_vec = flatten_model_update_cpu(model, start_snapshot, None)

    if top_vec is not None and update_vec.size == top_vec.size:
        proj = float(np.dot(update_vec.astype(np.float64), top_vec.astype(np.float64)))
        top_eig_proj_abs = abs(proj)
        top_eig_proj_ratio = float(abs(proj) / (float(np.linalg.norm(update_vec)) + 1e-12))

    return {
        "loss_before": float(loss_before),
        "loss_after": float(loss_after),
        "loss_delta": float(loss_after - loss_before),
        "mean_step_loss": float(np.mean(losses_during)) if losses_during else float("nan"),
        "first_grad_vec": first_grad_vec,
        "update_vec": update_vec,
        "top_eigval": float(top_eigval),
        "top_eig_proj_abs": float(top_eig_proj_abs),
        "top_eig_proj_ratio": float(top_eig_proj_ratio),
        **norm_stats,
    }


class MechanismClientALA(clientALA):
    def __init__(self, args, id, train_samples, test_samples):
        self.personalization_enabled = bool(getattr(args, "exp_personalization_enabled", True))
        self.freeze_mask = str(getattr(args, "exp_freeze_mask", "none"))
        self.local_momentum = float(getattr(args, "local_momentum", 0.0))
        self.local_weight_decay = float(getattr(args, "local_weight_decay", 0.0))
        self.grad_clip_norm = getattr(args, "grad_clip_norm", None)
        self.bn_policy = str(getattr(args, "bn_policy", "freeze_stats"))
        self.reset_ala_on_switch = bool(getattr(args, "reset_ala_on_switch", False))
        super().__init__(args, id, train_samples, test_samples)
        self._rebuild_optimizer()

    def _rebuild_optimizer(self):
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.learning_rate,
            momentum=self.local_momentum,
            weight_decay=self.local_weight_decay,
        )

    def _p_flags(self) -> List[bool]:
        return infer_p_flags(len(list(self.model.parameters())), int(self.layer_idx))

    def _frozen_flags(self) -> List[bool]:
        p_flags = self._p_flags()
        if self.freeze_mask == "none":
            return [False] * len(p_flags)
        if self.freeze_mask == "freeze_P":
            return p_flags
        if self.freeze_mask == "freeze_S":
            return [not x for x in p_flags]
        raise ValueError(f"Unsupported freeze mask: {self.freeze_mask}")

    def set_dataset(self, dataset: str, reset_optimizer: bool = False):
        self.dataset = dataset
        train_data = read_client_data(self.dataset, self.id, is_train=True)
        test_data = read_client_data(self.dataset, self.id, is_train=False)
        self.train_samples = len(train_data)
        self.test_samples = len(test_data)

        self.ALA.train_data = train_data
        if self.reset_ala_on_switch:
            self.ALA.weights = None
            self.ALA.start_phase = True

        if reset_optimizer:
            self._rebuild_optimizer()

    def local_initialization(self, received_global_model):
        if not self.personalization_enabled:
            for p, pg in zip(self.model.parameters(), received_global_model.parameters()):
                p.data = pg.data.clone()
            return
        super().local_initialization(received_global_model)

    def train(self):
        trainloader = self.load_train_data()
        self.model.train()
        apply_bn_policy(self.model, self.bn_policy)
        params = list(self.model.parameters())
        frozen_flags = self._frozen_flags()

        for _ in range(self.local_steps):
            for x, y in trainloader:
                x = move_x_to_device(x, self.device)
                y = y.to(self.device)
                self.optimizer.zero_grad()
                loss = self.loss(self.model(x), y)
                loss.backward()

                for p, frozen in zip(params, frozen_flags):
                    if frozen and p.grad is not None:
                        p.grad.zero_()

                if self.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(params, self.grad_clip_norm)

                self.optimizer.step()

    def run_probe_pair(
        self,
        global_model: torch.nn.Module,
        *,
        probe_steps: int,
        probe_batch_size: int,
        probe_seed: int,
        fixed_batch: bool,
        top_eig_enabled: bool,
        top_eig_power_steps: int,
    ) -> Dict[str, object]:
        probe_batches, batch_offset = get_probe_batches_from_client(
            client=self,
            steps=probe_steps,
            batch_size=probe_batch_size,
            seed=probe_seed,
            fixed_batch=fixed_batch,
        )
        if not probe_batches:
            return {
                "probe_status": "no_batches",
                "probe_batch_offset": int(batch_offset),
                "probe_batch_count": 0,
            }

        p_flags = self._p_flags()
        frozen_flags = self._frozen_flags()
        common = dict(
            device=self.device,
            loss_fn=self.loss,
            lr=self.learning_rate,
            momentum=self.local_momentum,
            weight_decay=self.local_weight_decay,
            grad_clip_norm=self.grad_clip_norm,
            frozen_flags=frozen_flags,
            p_flags=p_flags,
            top_eig_power_steps=top_eig_power_steps,
            top_eig_enabled=top_eig_enabled,
            bn_policy=self.bn_policy,
        )

        g_probe = run_short_sgd_probe(
            start_model=global_model,
            batches=probe_batches,
            **common,
        )
        p_probe = run_short_sgd_probe(
            start_model=self.model,
            batches=probe_batches,
            **common,
        )

        grad_alignment = float("nan")
        if g_probe["first_grad_vec"] is not None and p_probe["first_grad_vec"] is not None:
            grad_alignment = cosine_similarity_np(g_probe["first_grad_vec"], p_probe["first_grad_vec"])

        update_alignment = cosine_similarity_np(g_probe["update_vec"], p_probe["update_vec"])
        top_eig_proj_gap = float("nan")
        if not math.isnan(g_probe["top_eig_proj_abs"]) and not math.isnan(p_probe["top_eig_proj_abs"]):
            top_eig_proj_gap = float(p_probe["top_eig_proj_abs"] - g_probe["top_eig_proj_abs"])

        return {
            "probe_status": "ok",
            "probe_batch_offset": int(batch_offset),
            "probe_batch_count": len(probe_batches),
            "probe_seed": int(probe_seed),
            "probe_steps": int(probe_steps),
            "probe_batch_size": int(probe_batch_size),
            "grad_alignment": float(grad_alignment),
            "update_alignment": float(update_alignment),
            "risk_gap_loss_after_P_minus_G": float(p_probe["loss_after"] - g_probe["loss_after"]),
            "risk_gap_loss_delta_P_minus_G": float(p_probe["loss_delta"] - g_probe["loss_delta"]),
            "top_eig_proj_gap_P_minus_G": float(top_eig_proj_gap),
            "top_eig_enabled": int(top_eig_enabled),
            "G_loss_before": float(g_probe["loss_before"]),
            "G_loss_after": float(g_probe["loss_after"]),
            "G_loss_delta": float(g_probe["loss_delta"]),
            "G_mean_step_loss": float(g_probe["mean_step_loss"]),
            "G_U_norm": float(g_probe["U_norm"]),
            "G_U_P_norm": float(g_probe["U_P_norm"]),
            "G_U_S_norm": float(g_probe["U_S_norm"]),
            "G_U_P_ratio": float(g_probe["U_P_ratio"]),
            "G_top_eigval": float(g_probe["top_eigval"]),
            "G_top_eig_proj_abs": float(g_probe["top_eig_proj_abs"]),
            "G_top_eig_proj_ratio": float(g_probe["top_eig_proj_ratio"]),
            "P_loss_before": float(p_probe["loss_before"]),
            "P_loss_after": float(p_probe["loss_after"]),
            "P_loss_delta": float(p_probe["loss_delta"]),
            "P_mean_step_loss": float(p_probe["mean_step_loss"]),
            "P_U_norm": float(p_probe["U_norm"]),
            "P_U_P_norm": float(p_probe["U_P_norm"]),
            "P_U_S_norm": float(p_probe["U_S_norm"]),
            "P_U_P_ratio": float(p_probe["U_P_ratio"]),
            "P_top_eigval": float(p_probe["top_eigval"]),
            "P_top_eig_proj_abs": float(p_probe["top_eig_proj_abs"]),
            "P_top_eig_proj_ratio": float(p_probe["top_eig_proj_ratio"]),
        }


class ContinualMechanismFedALA:
    def __init__(self, args: argparse.Namespace, variant: ExperimentVariant, run_seed: int):
        self.args = args
        self.variant = variant
        self.run_seed = run_seed

        self.device = args.device
        self.task_datasets = list(args.task_datasets)
        self.rounds_per_task = int(args.rounds_per_task)
        self.total_rounds = len(self.task_datasets) * self.rounds_per_task

        self.num_clients = int(args.num_clients)
        self.join_ratio = float(args.join_ratio)
        self.join_clients = max(1, int(self.num_clients * self.join_ratio))
        self.random_join_ratio = bool(args.random_join_ratio)

        self.global_model = copy.deepcopy(args.model)

        self.clients: List[MechanismClientALA] = []
        self.selected_clients: List[MechanismClientALA] = []
        self.uploaded_weights: List[float] = []
        self.uploaded_models: List[torch.nn.Module] = []
        self.uploaded_ids: List[int] = []

        self.active_task_idx: Optional[int] = None
        self.active_dataset: Optional[str] = None

        self.last_eval_after: Dict[int, float] = {}
        self.round_rows: List[Dict[str, object]] = []
        self.probe_rows: List[Dict[str, object]] = []
        self.pers_rows: List[Dict[str, object]] = []
        self.local_eval_rows: List[Dict[str, object]] = []
        self.theory_rows: List[Dict[str, object]] = []
        self.task_best_global_acc: Dict[int, float] = {}
        self.task_best_global_params: Dict[int, List[torch.Tensor]] = {}

        # Fixed client sampling RNG, decoupled from training RNG for reproducible sampling.
        self.client_sampler = np.random.default_rng(args.sampling_seed + run_seed * 100_003)
        self.probe_sampler = np.random.default_rng(args.sampling_seed + run_seed * 200_003 + 17)
        self.local_eval_sampler = np.random.default_rng(args.sampling_seed + run_seed * 250_003 + 23)
        self.theory_sampler = np.random.default_rng(args.sampling_seed + run_seed * 300_003 + 31)

        self._set_clients()

    def _set_clients(self):
        dataset0 = self.task_datasets[0]
        for cid in range(self.num_clients):
            train_data = read_client_data(dataset0, cid, is_train=True)
            test_data = read_client_data(dataset0, cid, is_train=False)
            client = MechanismClientALA(
                self.args,
                id=cid,
                train_samples=len(train_data),
                test_samples=len(test_data),
            )
            self.clients.append(client)

    def set_active_task(self, task_idx: int):
        dataset = self.task_datasets[task_idx - 1]
        if self.active_task_idx == task_idx and self.active_dataset == dataset:
            return
        self.active_task_idx = task_idx
        self.active_dataset = dataset
        for c in self.clients:
            c.set_dataset(dataset, reset_optimizer=self.args.reset_optimizer_on_switch)

    def select_clients(self) -> List[MechanismClientALA]:
        if self.random_join_ratio:
            join_clients = int(self.client_sampler.integers(self.join_clients, self.num_clients + 1))
        else:
            join_clients = self.join_clients
        picked = self.client_sampler.choice(self.num_clients, size=join_clients, replace=False)
        return [self.clients[int(i)] for i in picked]

    def send_models(self):
        for c in self.clients:
            c.local_initialization(self.global_model)

    def receive_models(self):
        if not self.selected_clients:
            raise RuntimeError("No selected clients.")
        total_samples = sum(c.train_samples for c in self.selected_clients)
        if total_samples <= 0:
            raise RuntimeError("Selected clients have zero train samples.")

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
        for c in self.selected_clients:
            self.uploaded_weights.append(c.train_samples / total_samples)
            self.uploaded_ids.append(c.id)
            self.uploaded_models.append(c.model)

    def _add_parameters(self, w: float, client_model: torch.nn.Module):
        for sp, cp in zip(self.global_model.parameters(), client_model.parameters()):
            sp.data += cp.data.clone() * w

    def aggregate_parameters(self):
        if not self.uploaded_models:
            raise RuntimeError("No uploaded models.")
        self.global_model = copy.deepcopy(self.uploaded_models[0])
        for p in self.global_model.parameters():
            p.data = torch.zeros_like(p.data)
        for w, m in zip(self.uploaded_weights, self.uploaded_models):
            self._add_parameters(w, m)

    def evaluate_seen_tasks(self, task_indices: Iterable[int]) -> Dict[int, float]:
        out: Dict[int, float] = {}
        for t in task_indices:
            out[t] = evaluate_accuracy_on_dataset(
                self.global_model,
                self.task_datasets[t - 1],
                self.num_clients,
                self.args.batch_size,
                self.device,
            )
        return out

    def _round_update_stats(self, global_snapshot: Sequence[torch.Tensor]) -> Dict[str, float]:
        p_flags = infer_p_flags(len(global_snapshot), self.args.layer_idx)
        stats = [compute_update_norms(c.model, global_snapshot, p_flags) for c in self.selected_clients]
        return {
            "update_norm_mean": mean_or_nan([s["U_norm"] for s in stats]),
            "update_norm_std": std_or_nan([s["U_norm"] for s in stats]),
            "update_P_norm_mean": mean_or_nan([s["U_P_norm"] for s in stats]),
            "update_S_norm_mean": mean_or_nan([s["U_S_norm"] for s in stats]),
            "update_P_ratio_mean": mean_or_nan([s["U_P_ratio"] for s in stats]),
        }

    @staticmethod
    def _old_mean(eval_map: Dict[int, float], task_idx: int) -> float:
        vals = [float(v) for k, v in eval_map.items() if k < task_idx]
        return float(np.mean(vals)) if vals else float("nan")

    @staticmethod
    def _json_sorted_float_map(d: Dict[int, float]) -> str:
        return json.dumps({int(k): float(v) for k, v in sorted(d.items())}, ensure_ascii=True)

    def _update_task_best_checkpoint(self, task_idx: int, current_acc: float):
        best = self.task_best_global_acc.get(task_idx)
        if best is None or current_acc > best:
            self.task_best_global_acc[task_idx] = float(current_acc)
            self.task_best_global_params[task_idx] = snapshot_params_cpu(self.global_model)

    def _should_collect_theory(self, task_idx: int, round_in_task: int) -> bool:
        if not bool(self.args.theory_enable):
            return False
        if task_idx <= 1:
            return False
        window = int(self.args.theory_only_window_after_switch_rounds)
        if window <= 0:
            return True
        return round_in_task <= window

    def _theory_selected_clients(self) -> List[MechanismClientALA]:
        if not self.selected_clients:
            return []
        k = int(self.args.theory_clients_per_round)
        if k <= 0 or k >= len(self.selected_clients):
            return list(self.selected_clients)
        picked = self.theory_sampler.choice(len(self.selected_clients), size=k, replace=False)
        return [self.selected_clients[int(i)] for i in picked]

    def _local_eval_selected_clients(self) -> List[MechanismClientALA]:
        if not self.selected_clients:
            return []
        k = int(getattr(self.args, "local_eval_clients_per_round", 0))
        if k <= 0 or k >= len(self.selected_clients):
            return list(self.selected_clients)
        picked = self.local_eval_sampler.choice(len(self.selected_clients), size=k, replace=False)
        return [self.selected_clients[int(i)] for i in picked]

    def _collect_local_eval_metrics(
        self,
        *,
        global_round_idx: int,
        task_idx: int,
        round_in_task: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        if not bool(getattr(self.args, "log_local_eval_acc", 0)):
            return [], {}
        clients = self._local_eval_selected_clients()
        if not clients:
            return [], {}

        rows: List[Dict[str, object]] = []
        seen_tasks = list(range(1, task_idx + 1))

        buckets: Dict[str, List[float]] = {
            "current_g": [],
            "current_p": [],
            "current_d": [],
            "old_g": [],
            "old_p": [],
            "old_d": [],
            "seen_g": [],
            "seen_p": [],
            "seen_d": [],
        }
        weights: Dict[str, List[float]] = {
            "current": [],
            "old": [],
            "seen": [],
        }

        for client in clients:
            for eval_t in seen_tasks:
                eval_dataset = self.task_datasets[eval_t - 1]
                role = "current" if eval_t == task_idx else "old"
                status = "ok"
                g_acc = float("nan")
                p_acc = float("nan")
                n = 0
                try:
                    g_acc, n = evaluate_accuracy_on_client(
                        self.global_model,
                        eval_dataset,
                        client.id,
                        self.args.batch_size,
                        self.device,
                        is_train=False,
                    )
                    p_acc, _ = evaluate_accuracy_on_client(
                        client.model,
                        eval_dataset,
                        client.id,
                        self.args.batch_size,
                        self.device,
                        is_train=False,
                    )
                except ValueError:
                    status = "no_samples"

                delta = float(p_acc - g_acc) if status == "ok" else float("nan")
                rows.append(
                    {
                        "global_round": global_round_idx + 1,
                        "task_idx": task_idx,
                        "round_in_task": round_in_task,
                        "dataset": self.task_datasets[task_idx - 1],
                        "client_id": client.id,
                        "eval_task_idx": eval_t,
                        "eval_dataset": eval_dataset,
                        "eval_task_role": role,
                        "eval_stage": "pre_local",
                        "exp_id": self.variant.exp_id,
                        "run_seed": self.run_seed,
                        "global_local_acc": float(g_acc),
                        "personalized_local_acc": float(p_acc),
                        "delta_local_acc": float(delta),
                        "test_samples": int(n),
                        "eval_status": status,
                    }
                )

                if status != "ok" or math.isnan(g_acc) or math.isnan(p_acc):
                    continue

                w = float(n)
                buckets["seen_g"].append(float(g_acc))
                buckets["seen_p"].append(float(p_acc))
                buckets["seen_d"].append(float(delta))
                weights["seen"].append(w)

                if role == "current":
                    buckets["current_g"].append(float(g_acc))
                    buckets["current_p"].append(float(p_acc))
                    buckets["current_d"].append(float(delta))
                    weights["current"].append(w)
                else:
                    buckets["old_g"].append(float(g_acc))
                    buckets["old_p"].append(float(p_acc))
                    buckets["old_d"].append(float(delta))
                    weights["old"].append(w)

        def _pack(split: str, short: str, values: List[float]) -> Dict[str, float]:
            out = {
                f"local_eval_{short}_{split}_mean": mean_or_nan(values),
                f"local_eval_{short}_{split}_std": std_or_nan(values),
            }
            if short in ("global_acc", "personalized_acc", "delta_acc"):
                weight_key = "current" if split == "current" else ("old" if split == "old" else "seen")
                out[f"local_eval_{short}_{split}_wmean"] = weighted_mean_or_nan(values, weights[weight_key])
            return out

        aggs: Dict[str, float] = {}
        aggs.update(_pack("current", "global_acc", buckets["current_g"]))
        aggs.update(_pack("current", "personalized_acc", buckets["current_p"]))
        aggs.update(_pack("current", "delta_acc", buckets["current_d"]))
        aggs.update(_pack("old", "global_acc", buckets["old_g"]))
        aggs.update(_pack("old", "personalized_acc", buckets["old_p"]))
        aggs.update(_pack("old", "delta_acc", buckets["old_d"]))
        aggs.update(_pack("seen", "global_acc", buckets["seen_g"]))
        aggs.update(_pack("seen", "personalized_acc", buckets["seen_p"]))
        aggs.update(_pack("seen", "delta_acc", buckets["seen_d"]))
        aggs["local_eval_num_clients"] = int(len(clients))
        aggs["local_eval_num_task_evals"] = int(sum(1 for r in rows if str(r.get("eval_status")) == "ok"))
        return rows, aggs

    def _collect_personalization_gain(
        self,
        *,
        global_round_idx: int,
        task_idx: int,
        round_in_task: int,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        if not bool(self.args.log_personalization_gain):
            return [], {"delta_pers_old_mean": float("nan"), "delta_pers_old_std": float("nan")}
        if task_idx <= 1:
            return [], {"delta_pers_old_mean": float("nan"), "delta_pers_old_std": float("nan")}

        clients = list(self.selected_clients)
        k = int(self.args.pers_clients_per_round)
        if k > 0 and k < len(clients):
            picked = self.theory_sampler.choice(len(clients), size=k, replace=False)
            clients = [clients[int(i)] for i in picked]

        rows: List[Dict[str, object]] = []
        deltas: List[float] = []
        for client in clients:
            old_task_deltas: List[float] = []
            for old_t in range(1, task_idx):
                dataset_old = self.task_datasets[old_t - 1]
                g_acc, n = evaluate_accuracy_on_client(
                    self.global_model,
                    dataset_old,
                    client.id,
                    self.args.batch_size,
                    self.device,
                    is_train=False,
                )
                p_acc, _ = evaluate_accuracy_on_client(
                    client.model,
                    dataset_old,
                    client.id,
                    self.args.batch_size,
                    self.device,
                    is_train=False,
                )
                delta = float(p_acc - g_acc)
                rows.append(
                    {
                        "global_round": global_round_idx + 1,
                        "task_idx": task_idx,
                        "round_in_task": round_in_task,
                        "dataset": self.task_datasets[task_idx - 1],
                        "client_id": client.id,
                        "old_task_idx": old_t,
                        "old_dataset": dataset_old,
                        "exp_id": self.variant.exp_id,
                        "run_seed": self.run_seed,
                        "global_acc_old_task": float(g_acc),
                        "personalized_acc_old_task": float(p_acc),
                        "delta_pers": delta,
                        "test_samples": int(n),
                    }
                )
                old_task_deltas.append(delta)
                deltas.append(delta)
            if old_task_deltas:
                rows.append(
                    {
                        "global_round": global_round_idx + 1,
                        "task_idx": task_idx,
                        "round_in_task": round_in_task,
                        "dataset": self.task_datasets[task_idx - 1],
                        "client_id": client.id,
                        "old_task_idx": 0,
                        "old_dataset": "OLD_TASKS_MEAN",
                        "exp_id": self.variant.exp_id,
                        "run_seed": self.run_seed,
                        "global_acc_old_task": float("nan"),
                        "personalized_acc_old_task": float("nan"),
                        "delta_pers": float(np.mean(old_task_deltas)),
                        "test_samples": 0,
                    }
                )

        return rows, {
            "delta_pers_old_mean": mean_or_nan(deltas),
            "delta_pers_old_std": std_or_nan(deltas),
        }

    def _theory_lambda_for_model_client_task(
        self,
        model: torch.nn.Module,
        *,
        dataset: str,
        client_id: int,
        seed: int,
    ) -> Tuple[float, int]:
        batch, batch_offset = get_fixed_batch_for_dataset_client(
            dataset=dataset,
            client_id=client_id,
            batch_size=int(self.args.theory_batch_size),
            seed=seed,
            is_train=True,
        )
        if batch is None:
            return float("nan"), int(batch_offset)
        eig, _ = _hessian_top_eig_first_batch(
            model=copy.deepcopy(model).to(self.device),
            batch=batch,
            loss_fn=torch.nn.CrossEntropyLoss(),
            device=self.device,
            power_steps=int(self.args.theory_lambda_power_steps),
            bn_policy=str(self.args.bn_policy),
        )
        return float(eig), int(batch_offset)

    def _collect_theory_metrics(
        self,
        *,
        phase: str,
        global_round_idx: int,
        task_idx: int,
        round_in_task: int,
        global_model_ref: torch.nn.Module,
        client_models: Optional[Sequence[Tuple[int, torch.nn.Module]]] = None,
        include_global: bool = True,
    ) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
        if not self._should_collect_theory(task_idx, round_in_task):
            return [], {}
        old_tasks = [t for t in range(1, task_idx) if t in self.task_best_global_params]
        if not old_tasks:
            return [], {}

        rows: List[Dict[str, object]] = []
        agg: Dict[str, List[float]] = {
            "lambda_global": [],
            "opt_drift_global": [],
            "lambda_client": [],
            "opt_drift_client": [],
        }

        if include_global and (bool(self.args.theory_lambda_enable) or bool(self.args.theory_opt_drift_enable)):
            # Global measurements over sampled clients, aggregated across old tasks.
            if int(self.args.theory_global_clients_per_task) <= 0:
                global_client_ids = list(range(self.num_clients))
            else:
                k_global = min(self.num_clients, int(self.args.theory_global_clients_per_task))
                global_client_ids = self.theory_sampler.choice(self.num_clients, size=k_global, replace=False).tolist()

            for old_t in old_tasks:
                dataset_old = self.task_datasets[old_t - 1]
                # Average lambda_max over sampled clients for stability.
                lambda_vals: List[float] = []
                batch_offsets: List[int] = []
                if bool(self.args.theory_lambda_enable):
                    for cid in global_client_ids:
                        seed = int(
                            self.args.theory_fixed_batch_seed
                            + (global_round_idx + 1) * 10_001
                            + old_t * 1_009
                            + cid * 97
                        )
                        lam, off = self._theory_lambda_for_model_client_task(
                            global_model_ref,
                            dataset=dataset_old,
                            client_id=int(cid),
                            seed=seed,
                        )
                        if not math.isnan(lam):
                            lambda_vals.append(float(lam))
                        batch_offsets.append(int(off))
                lambda_mean = mean_or_nan(lambda_vals) if lambda_vals else float("nan")

                opt_drift_val = float("nan")
                if bool(self.args.theory_opt_drift_enable):
                    opt_drift_val = param_l2_distance(global_model_ref, self.task_best_global_params[old_t])

                rows.append(
                    {
                        "global_round": global_round_idx + 1,
                        "task_idx": task_idx,
                        "round_in_task": round_in_task,
                        "phase": phase,
                        "scope": "global",
                        "client_id": -1,
                        "old_task_idx": old_t,
                        "old_dataset": dataset_old,
                        "exp_id": self.variant.exp_id,
                        "run_seed": self.run_seed,
                        "lambda_max": float(lambda_mean),
                        "opt_drift": float(opt_drift_val),
                        "batch_offsets": ";".join(str(o) for o in batch_offsets),
                        "num_clients_used": len(global_client_ids),
                    }
                )
                if not math.isnan(lambda_mean):
                    agg["lambda_global"].append(float(lambda_mean))
                if not math.isnan(opt_drift_val):
                    agg["opt_drift_global"].append(float(opt_drift_val))

        if client_models and (bool(self.args.theory_lambda_enable) or bool(self.args.theory_opt_drift_enable)):
            for cid, model_ref in client_models:
                for old_t in old_tasks:
                    dataset_old = self.task_datasets[old_t - 1]
                    lambda_val = float("nan")
                    batch_offset = -1
                    if bool(self.args.theory_lambda_enable):
                        seed = int(
                            self.args.theory_fixed_batch_seed
                            + (global_round_idx + 1) * 20_011
                            + old_t * 1_313
                            + cid * 131
                        )
                        lambda_val, batch_offset = self._theory_lambda_for_model_client_task(
                            model_ref,
                            dataset=dataset_old,
                            client_id=cid,
                            seed=seed,
                        )

                    opt_drift_val = float("nan")
                    if bool(self.args.theory_opt_drift_enable):
                        opt_drift_val = param_l2_distance(model_ref, self.task_best_global_params[old_t])

                    rows.append(
                        {
                            "global_round": global_round_idx + 1,
                            "task_idx": task_idx,
                            "round_in_task": round_in_task,
                            "phase": phase,
                            "scope": "client",
                            "client_id": int(cid),
                            "old_task_idx": old_t,
                            "old_dataset": dataset_old,
                            "exp_id": self.variant.exp_id,
                            "run_seed": self.run_seed,
                            "lambda_max": float(lambda_val),
                            "opt_drift": float(opt_drift_val),
                            "batch_offsets": str(batch_offset),
                            "num_clients_used": 1,
                        }
                    )
                    if not math.isnan(lambda_val):
                        agg["lambda_client"].append(float(lambda_val))
                    if not math.isnan(opt_drift_val):
                        agg["opt_drift_client"].append(float(opt_drift_val))

        aggregates = {}
        if agg["lambda_global"]:
            aggregates[f"{phase}_lambda_max_global_old_mean"] = float(np.mean(agg["lambda_global"]))
        if agg["opt_drift_global"]:
            aggregates[f"{phase}_opt_drift_global_old_mean"] = float(np.mean(agg["opt_drift_global"]))
        if agg["lambda_client"]:
            aggregates[f"{phase}_lambda_max_client_old_mean"] = float(np.mean(agg["lambda_client"]))
        if agg["opt_drift_client"]:
            aggregates[f"{phase}_opt_drift_client_old_mean"] = float(np.mean(agg["opt_drift_client"]))
        return rows, aggregates

    def _run_probe_for_round(
        self,
        *,
        global_round_idx: int,
        task_idx: int,
        round_in_task: int,
    ) -> List[Dict[str, object]]:
        if not self.variant.probe_enabled:
            return []
        if task_idx <= 1:
            return []
        if round_in_task > self.args.probe_only_window_after_switch_rounds:
            return []
        if self.args.probe_steps <= 0 or self.args.probe_clients_per_round <= 0:
            return []
        if not self.selected_clients:
            return []

        n_probe = min(len(self.selected_clients), self.args.probe_clients_per_round)
        if n_probe <= 0:
            return []
        picked_idx = self.probe_sampler.choice(len(self.selected_clients), size=n_probe, replace=False)

        rows: List[Dict[str, object]] = []
        for local_rank, idx in enumerate(picked_idx.tolist()):
            client = self.selected_clients[int(idx)]
            probe_seed = int(
                self.args.probe_fixed_seed
                + self.run_seed * 1_000_003
                + (global_round_idx + 1) * 10_007
                + client.id * 97
                + local_rank
            )
            probe_metrics = client.run_probe_pair(
                global_model=self.global_model,
                probe_steps=self.args.probe_steps,
                probe_batch_size=self.args.probe_batch_size or self.args.batch_size,
                probe_seed=probe_seed,
                fixed_batch=bool(self.args.probe_fixed_batch),
                top_eig_enabled=bool(self.args.probe_enable_top_eig),
                top_eig_power_steps=self.args.probe_top_eig_power_steps,
            )
            rows.append(
                {
                    "global_round": global_round_idx + 1,
                    "task_idx": task_idx,
                    "round_in_task": round_in_task,
                    "dataset": self.task_datasets[task_idx - 1],
                    "client_id": client.id,
                    "exp_id": self.variant.exp_id,
                    "run_seed": self.run_seed,
                    **probe_metrics,
                }
            )
        return rows

    def run(
        self,
    ) -> Tuple[
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        List[Dict[str, object]],
        Dict[str, object],
    ]:
        if not self.task_datasets:
            raise ValueError("task_datasets is empty.")

        self.set_active_task(1)
        self.last_eval_after = self.evaluate_seen_tasks([1])

        for r in range(self.total_rounds):
            task_idx = r // self.rounds_per_task + 1
            round_in_task = r % self.rounds_per_task + 1
            is_switch_round = int(task_idx > 1 and round_in_task == 1)
            is_switch_window = int(task_idx > 1 and round_in_task <= self.args.probe_only_window_after_switch_rounds)

            if task_idx != self.active_task_idx:
                self.set_active_task(task_idx)
                # Carry old-task cached evals and add current task's pre-round accuracy.
                self.last_eval_after[task_idx] = evaluate_accuracy_on_dataset(
                    self.global_model,
                    self.task_datasets[task_idx - 1],
                    self.num_clients,
                    self.args.batch_size,
                    self.device,
                )

            seen_tasks = list(range(1, task_idx + 1))
            before_eval = {k: self.last_eval_after[k] for k in seen_tasks}
            global_snapshot = snapshot_params_cpu(self.global_model)

            self.selected_clients = self.select_clients()
            self.send_models()

            local_eval_rows_this_round, local_eval_aggs = self._collect_local_eval_metrics(
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
            )
            self.local_eval_rows.extend(local_eval_rows_this_round)

            pers_rows_this_round, pers_aggs = self._collect_personalization_gain(
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
            )
            self.pers_rows.extend(pers_rows_this_round)

            theory_client_refs_pre: Optional[List[Tuple[int, torch.nn.Module]]] = None
            if bool(self.args.theory_include_personalized):
                theory_clients = self._theory_selected_clients()
                theory_client_refs_pre = [(c.id, c.model) for c in theory_clients]
            theory_rows_pre, theory_aggs_pre = self._collect_theory_metrics(
                phase="before_local",
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
                global_model_ref=self.global_model,
                client_models=theory_client_refs_pre,
            )
            self.theory_rows.extend(theory_rows_pre)

            probe_rows_this_round = self._run_probe_for_round(
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
            )
            self.probe_rows.extend(probe_rows_this_round)
            for c in self.selected_clients:
                c.train()

            update_stats = self._round_update_stats(global_snapshot)

            theory_clients_post = self._theory_selected_clients()
            theory_rows_local, theory_aggs_local = self._collect_theory_metrics(
                phase="after_local",
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
                global_model_ref=self.global_model,
                client_models=[(c.id, c.model) for c in theory_clients_post],
                include_global=False,
            )
            self.theory_rows.extend(theory_rows_local)

            self.receive_models()
            self.aggregate_parameters()

            theory_rows_postagg, theory_aggs_postagg = self._collect_theory_metrics(
                phase="after_agg",
                global_round_idx=r,
                task_idx=task_idx,
                round_in_task=round_in_task,
                global_model_ref=self.global_model,
                client_models=None,
            )
            self.theory_rows.extend(theory_rows_postagg)

            after_eval = self.evaluate_seen_tasks(seen_tasks)
            self.last_eval_after = dict(after_eval)

            delta_agg = {k: float(after_eval[k] - before_eval[k]) for k in seen_tasks}

            row = {
                "global_round": r + 1,
                "task_idx": task_idx,
                "round_in_task": round_in_task,
                "dataset": self.task_datasets[task_idx - 1],
                "is_task_switch_round": is_switch_round,
                "is_switch_window_round": is_switch_window,
                "num_selected_clients": len(self.selected_clients),
                "selected_client_ids": ";".join(str(c.id) for c in self.selected_clients),
                "personalization_enabled": int(self.variant.personalization_enabled),
                "freeze_mask": self.variant.freeze_mask,
                "probe_enabled_flag": int(self.variant.probe_enabled),
                "probe_rows_logged": len(probe_rows_this_round),
                "local_eval_rows_logged": len(local_eval_rows_this_round),
                "acc_current_before": float(before_eval[task_idx]),
                "acc_current_after": float(after_eval[task_idx]),
                "delta_agg_current": float(delta_agg[task_idx]),
                "acc_old_mean_before": self._old_mean(before_eval, task_idx),
                "acc_old_mean_after": self._old_mean(after_eval, task_idx),
                "delta_agg_old_mean": self._old_mean(delta_agg, task_idx),
                "delta_pers_old_mean": float(pers_aggs.get("delta_pers_old_mean", float("nan"))),
                "delta_pers_old_std": float(pers_aggs.get("delta_pers_old_std", float("nan"))),
                "pers_rows_logged": len(pers_rows_this_round),
                "acc_before_json": self._json_sorted_float_map(before_eval),
                "acc_after_json": self._json_sorted_float_map(after_eval),
                "delta_agg_json": self._json_sorted_float_map(delta_agg),
            }
            local_eval_keys = [
                "local_eval_global_acc_current_mean",
                "local_eval_global_acc_current_std",
                "local_eval_global_acc_current_wmean",
                "local_eval_personalized_acc_current_mean",
                "local_eval_personalized_acc_current_std",
                "local_eval_personalized_acc_current_wmean",
                "local_eval_delta_acc_current_mean",
                "local_eval_delta_acc_current_std",
                "local_eval_delta_acc_current_wmean",
                "local_eval_global_acc_old_mean",
                "local_eval_global_acc_old_std",
                "local_eval_global_acc_old_wmean",
                "local_eval_personalized_acc_old_mean",
                "local_eval_personalized_acc_old_std",
                "local_eval_personalized_acc_old_wmean",
                "local_eval_delta_acc_old_mean",
                "local_eval_delta_acc_old_std",
                "local_eval_delta_acc_old_wmean",
                "local_eval_global_acc_seen_mean",
                "local_eval_global_acc_seen_std",
                "local_eval_global_acc_seen_wmean",
                "local_eval_personalized_acc_seen_mean",
                "local_eval_personalized_acc_seen_std",
                "local_eval_personalized_acc_seen_wmean",
                "local_eval_delta_acc_seen_mean",
                "local_eval_delta_acc_seen_std",
                "local_eval_delta_acc_seen_wmean",
                "local_eval_num_clients",
                "local_eval_num_task_evals",
            ]
            for k in local_eval_keys:
                row[k] = float(local_eval_aggs.get(k, float("nan"))) if "num_" not in k else int(
                    local_eval_aggs.get(k, 0)
                )
            if probe_rows_this_round:
                row["probe_grad_alignment_mean"] = mean_or_nan(
                    [float(pr["grad_alignment"]) for pr in probe_rows_this_round if str(pr.get("probe_status")) == "ok"]
                )
                row["probe_update_alignment_mean"] = mean_or_nan(
                    [float(pr["update_alignment"]) for pr in probe_rows_this_round if str(pr.get("probe_status")) == "ok"]
                )
                row["probe_top_eig_proj_gap_mean"] = mean_or_nan(
                    [float(pr["top_eig_proj_gap_P_minus_G"]) for pr in probe_rows_this_round if str(pr.get("probe_status")) == "ok"]
                )
                row["probe_risk_gap_loss_after_mean"] = mean_or_nan(
                    [float(pr["risk_gap_loss_after_P_minus_G"]) for pr in probe_rows_this_round if str(pr.get("probe_status")) == "ok"]
                )
            else:
                row["probe_grad_alignment_mean"] = float("nan")
                row["probe_update_alignment_mean"] = float("nan")
                row["probe_top_eig_proj_gap_mean"] = float("nan")
                row["probe_risk_gap_loss_after_mean"] = float("nan")
            # Ensure all theory aggregate columns exist for every round.
            theory_all = {}
            theory_all.update(theory_aggs_pre)
            theory_all.update(theory_aggs_local)
            theory_all.update(theory_aggs_postagg)
            theory_keys = [
                "before_local_lambda_max_global_old_mean",
                "before_local_opt_drift_global_old_mean",
                "before_local_lambda_max_client_old_mean",
                "before_local_opt_drift_client_old_mean",
                "after_local_lambda_max_global_old_mean",
                "after_local_opt_drift_global_old_mean",
                "after_local_lambda_max_client_old_mean",
                "after_local_opt_drift_client_old_mean",
                "after_agg_lambda_max_global_old_mean",
                "after_agg_opt_drift_global_old_mean",
                "after_agg_lambda_max_client_old_mean",
                "after_agg_opt_drift_client_old_mean",
            ]
            for k in theory_keys:
                row[k] = float(theory_all.get(k, float("nan")))
            row["theory_rows_logged"] = len(theory_rows_pre) + len(theory_rows_local) + len(theory_rows_postagg)
            row.update(update_stats)
            self.round_rows.append(row)

            old_after = row["acc_old_mean_after"]
            d_agg_old = row["delta_agg_old_mean"]
            old_after_str = f"{old_after:.4f}" if not (isinstance(old_after, float) and math.isnan(old_after)) else "nan"
            d_agg_str = f"{d_agg_old:.4f}" if not (isinstance(d_agg_old, float) and math.isnan(d_agg_old)) else "nan"
            print(
                f"[{self.variant.exp_id}][seed={self.run_seed}] "
                f"round {r + 1}/{self.total_rounds} task={task_idx} r={round_in_task} "
                f"old_after={old_after_str} dAgg_old={d_agg_str} "
                f"U_P={row['update_P_norm_mean']:.4f} "
                f"probe_n={row['probe_rows_logged']} "
                f"local_eval_n={row['local_eval_rows_logged']} "
                f"pers_n={row['pers_rows_logged']} "
                f"theory_n={row['theory_rows_logged']}"
            )

            self._update_task_best_checkpoint(task_idx, float(after_eval[task_idx]))

        summary = self.build_summary()
        return self.round_rows, self.probe_rows, self.pers_rows, self.local_eval_rows, self.theory_rows, summary

    def build_summary(self) -> Dict[str, object]:
        rows = self.round_rows
        if not rows:
            return {}

        final = rows[-1]
        old_shocks = [
            float(r["delta_agg_old_mean"])
            for r in rows
            if not (isinstance(r["delta_agg_old_mean"], float) and math.isnan(r["delta_agg_old_mean"]))
        ]
        switch_window_shocks = [
            float(r["delta_agg_old_mean"])
            for r in rows
            if int(r["is_switch_window_round"]) == 1
            and not (isinstance(r["delta_agg_old_mean"], float) and math.isnan(r["delta_agg_old_mean"]))
        ]
        probe_ok = [r for r in self.probe_rows if str(r.get("probe_status")) == "ok"]
        pers_vals = []
        for r in self.pers_rows:
            try:
                if int(r.get("old_task_idx", -1)) == 0:
                    v = float(r.get("delta_pers"))
                    if not math.isnan(v):
                        pers_vals.append(v)
            except Exception:
                continue
        theory_lambda_global = []
        theory_opt_global = []
        theory_lambda_client = []
        theory_opt_client = []
        for tr in self.theory_rows:
            try:
                lam = float(tr.get("lambda_max"))
                if not math.isnan(lam):
                    if str(tr.get("scope")) == "global":
                        theory_lambda_global.append(lam)
                    else:
                        theory_lambda_client.append(lam)
            except Exception:
                pass
            try:
                od = float(tr.get("opt_drift"))
                if not math.isnan(od):
                    if str(tr.get("scope")) == "global":
                        theory_opt_global.append(od)
                    else:
                        theory_opt_client.append(od)
            except Exception:
                pass

        def _probe_mean(key: str):
            vals = []
            for r in probe_ok:
                try:
                    v = float(r.get(key))
                except Exception:
                    continue
                if math.isnan(v):
                    continue
                vals.append(v)
            return float(np.mean(vals)) if vals else None

        return {
            "variant": asdict(self.variant),
            "seed": self.run_seed,
            "task_datasets": list(self.task_datasets),
            "rounds_per_task": self.rounds_per_task,
            "total_rounds": self.total_rounds,
            "final_acc_current_after": float(final["acc_current_after"]),
            "final_acc_old_mean_after": None
            if (isinstance(final["acc_old_mean_after"], float) and math.isnan(final["acc_old_mean_after"]))
            else float(final["acc_old_mean_after"]),
            "min_delta_agg_old_mean": float(min(old_shocks)) if old_shocks else None,
            "mean_delta_agg_old_mean": float(np.mean(old_shocks)) if old_shocks else None,
            "mean_delta_agg_old_mean_switch_window": float(np.mean(switch_window_shocks))
            if switch_window_shocks
            else None,
            "max_update_P_norm_mean": float(max(float(r["update_P_norm_mean"]) for r in rows)),
            "max_update_S_norm_mean": float(max(float(r["update_S_norm_mean"]) for r in rows)),
            "probe_rows": len(self.probe_rows),
            "probe_enabled": bool(self.variant.probe_enabled),
            "probe_summary": {
                "ok_rows": len(probe_ok),
                "mean_grad_alignment": _probe_mean("grad_alignment"),
                "mean_update_alignment": _probe_mean("update_alignment"),
                "mean_top_eig_proj_gap_P_minus_G": _probe_mean("top_eig_proj_gap_P_minus_G"),
                "mean_risk_gap_loss_after_P_minus_G": _probe_mean("risk_gap_loss_after_P_minus_G"),
            },
            "pers_rows": len(self.pers_rows),
            "local_eval_rows": len(self.local_eval_rows),
            "local_eval_summary": {
                "final_local_eval_global_acc_current_mean": None
                if (isinstance(final.get("local_eval_global_acc_current_mean"), float) and math.isnan(final["local_eval_global_acc_current_mean"]))
                else final.get("local_eval_global_acc_current_mean"),
                "final_local_eval_personalized_acc_current_mean": None
                if (isinstance(final.get("local_eval_personalized_acc_current_mean"), float) and math.isnan(final["local_eval_personalized_acc_current_mean"]))
                else final.get("local_eval_personalized_acc_current_mean"),
                "final_local_eval_global_acc_old_mean": None
                if (isinstance(final.get("local_eval_global_acc_old_mean"), float) and math.isnan(final["local_eval_global_acc_old_mean"]))
                else final.get("local_eval_global_acc_old_mean"),
                "final_local_eval_personalized_acc_old_mean": None
                if (isinstance(final.get("local_eval_personalized_acc_old_mean"), float) and math.isnan(final["local_eval_personalized_acc_old_mean"]))
                else final.get("local_eval_personalized_acc_old_mean"),
                "final_local_eval_global_acc_seen_mean": None
                if (isinstance(final.get("local_eval_global_acc_seen_mean"), float) and math.isnan(final["local_eval_global_acc_seen_mean"]))
                else final.get("local_eval_global_acc_seen_mean"),
                "final_local_eval_personalized_acc_seen_mean": None
                if (isinstance(final.get("local_eval_personalized_acc_seen_mean"), float) and math.isnan(final["local_eval_personalized_acc_seen_mean"]))
                else final.get("local_eval_personalized_acc_seen_mean"),
            },
            "pers_summary": {
                "mean_delta_pers_old_mean_over_client_rounds": float(np.mean(pers_vals)) if pers_vals else None,
                "max_delta_pers_old_mean_over_client_rounds": float(np.max(pers_vals)) if pers_vals else None,
            },
            "theory_rows": len(self.theory_rows),
            "theory_summary": {
                "mean_lambda_max_global": float(np.mean(theory_lambda_global)) if theory_lambda_global else None,
                "mean_opt_drift_global": float(np.mean(theory_opt_global)) if theory_opt_global else None,
                "mean_lambda_max_client": float(np.mean(theory_lambda_client)) if theory_lambda_client else None,
                "mean_opt_drift_client": float(np.mean(theory_opt_client)) if theory_opt_client else None,
            },
            "unsupported": {
                "theory_notes": "lambda_max is estimated via power-iteration HVP on fixed train batches; opt_drift uses best global checkpoint per task",
            },
        }


def build_runner_args(parsed: argparse.Namespace, model: torch.nn.Module, variant: ExperimentVariant):
    args = copy.deepcopy(parsed)
    args.model = model
    args.dataset = parsed.task_datasets[0]
    args.exp_personalization_enabled = variant.personalization_enabled
    args.exp_freeze_mask = variant.freeze_mask
    # Legacy field names expected by clientALA.
    args.local_learning_rate = parsed.local_lr
    return args


def config_snapshot(parsed: argparse.Namespace, variant: ExperimentVariant) -> Dict[str, object]:
    return {
        "experiment": {
            "run_name": variant.run_name,
            "seed": parsed.seed,
            "times": parsed.times,
            "output_dir": parsed.output_dir,
        },
        "task_sequence": {
            "datasets": list(parsed.task_datasets),
            "num_tasks": len(parsed.task_datasets),
            "rounds_per_task": parsed.rounds_per_task,
            "reset_optimizer_on_switch": bool(parsed.reset_optimizer_on_switch),
            "reset_model_on_switch": False,
        },
        "federated": {
            "num_clients": parsed.num_clients,
            "client_fraction": parsed.join_ratio,
            "sampling_seed": parsed.sampling_seed,
            "aggregator": "fedavg",
        },
        "local_train": {
            "optimizer": "sgd",
            "lr": parsed.local_lr,
            "momentum": parsed.local_momentum,
            "weight_decay": parsed.local_weight_decay,
            "batch_size": parsed.batch_size,
            "local_epochs_or_steps": parsed.local_steps,
            "grad_clip_norm": parsed.grad_clip_norm,
        },
        "personalization": {
            "method": "fedala",
            "enabled": variant.personalization_enabled,
            "rand_percent": parsed.rand_percent,
            "eta": parsed.eta,
            "layer_idx": parsed.layer_idx,
            "ala_threshold": parsed.ala_threshold,
            "ala_num_pre_loss": parsed.ala_num_pre_loss,
        },
        "freeze": {
            "enabled": variant.freeze_mask != "none",
            "mask": variant.freeze_mask,
            "bn_policy": parsed.bn_policy,
        },
        "mechanism_logging": {
            "log_update_norms": True,
            "log_update_subspace_norms": True,
            "log_step_shocks": True,
            "log_local_eval_acc": bool(parsed.log_local_eval_acc),
            "local_eval_clients_per_round": (
                "all_participants" if parsed.local_eval_clients_per_round <= 0 else parsed.local_eval_clients_per_round
            ),
            "local_eval_eval_tasks": "seen_tasks",
            "local_eval_eval_split": "test",
            "local_eval_stage": "pre_local",
            "log_personalization_gain": bool(parsed.log_personalization_gain),
            "record_clients_per_round": (
                "all_participants" if parsed.pers_clients_per_round <= 0 else parsed.pers_clients_per_round
            ),
        },
        "probe": {
            "enabled": variant.probe_enabled,
            "placeholder_only": False,
            "only_window_after_switch_rounds": parsed.probe_only_window_after_switch_rounds,
            "clients_per_round": parsed.probe_clients_per_round,
            "steps": parsed.probe_steps,
            "batch_size": parsed.probe_batch_size or parsed.batch_size,
            "fixed_seed": parsed.probe_fixed_seed,
            "fixed_batch": bool(parsed.probe_fixed_batch),
            "record": [
                "U_norm",
                "U_P_norm",
                "U_S_norm",
                "grad_alignment",
                "update_alignment",
                "top_eig_proj",
                "risk_gap",
            ],
            "top_eig": {
                "enabled": bool(parsed.probe_enable_top_eig),
                "power_steps": parsed.probe_top_eig_power_steps,
                "batch_source": "first_probe_batch",
            },
        },
        "theory_metrics": {
            "enabled": bool(parsed.theory_enable),
            "lambda_max": {
                "enabled": bool(parsed.theory_lambda_enable),
                "method": "power_iter_hvp_fixed_batch",
                "power_steps": parsed.theory_lambda_power_steps,
                "batch_size": parsed.theory_batch_size,
                "fixed_batch_seed": parsed.theory_fixed_batch_seed,
                "eval_tasks": "old_tasks_mean",
            },
            "opt_drift": {
                "enabled": bool(parsed.theory_opt_drift_enable),
                "mode": "best_checkpoint_per_task",
                "distance": "l2",
            },
            "window_after_switch_rounds": parsed.theory_only_window_after_switch_rounds,
            "global_clients_per_task": parsed.theory_global_clients_per_task,
            "client_models_per_round": parsed.theory_clients_per_round,
            "include_personalized": bool(parsed.theory_include_personalized),
        },
        "runtime": {
            "device": parsed.device,
            "device_id": parsed.device_id,
            "model": parsed.model_name,
            "num_classes": parsed.num_classes,
        },
    }


def summarize_suite(rows: List[Dict[str, object]]) -> Dict[str, object]:
    grouped: Dict[str, List[Dict[str, object]]] = {}
    for r in rows:
        grouped.setdefault(str(r["exp_id"]), []).append(r)

    out: Dict[str, object] = {"num_runs": len(rows), "experiments": {}}
    for exp_id, items in grouped.items():
        def vals(key: str) -> List[float]:
            result = []
            for item in items:
                v = item.get(key)
                if v is None:
                    continue
                try:
                    fv = float(v)
                except Exception:
                    continue
                if math.isnan(fv):
                    continue
                result.append(fv)
            return result

        final_old = vals("final_acc_old_mean_after")
        final_cur = vals("final_acc_current_after")
        min_shock = vals("min_delta_agg_old_mean")
        out["experiments"][exp_id] = {
            "runs": len(items),
            "final_acc_old_mean_after_mean": float(np.mean(final_old)) if final_old else None,
            "final_acc_current_after_mean": float(np.mean(final_cur)) if final_cur else None,
            "min_delta_agg_old_mean_mean": float(np.mean(min_shock)) if min_shock else None,
        }
    return out


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "A/B/C/D FedALA continual experiment runner based on 实验设置.md "
            "(Full-FedALA, No-Pers, Freeze-P, Freeze-S)."
        )
    )
    p.add_argument("--task_datasets", type=str, required=True)
    p.add_argument("--rounds_per_task", type=int, default=20)
    p.add_argument("--experiments", type=str, default="A,B,C,D")
    p.add_argument("--times", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
    p.add_argument("--device_id", type=str, default="0")
    p.add_argument("--model", dest="model_name", type=str, default="cnn", choices=["cnn", "resnet", "fastText"])
    p.add_argument("--num_classes", type=int, default=10)

    p.add_argument("--num_clients", type=int, default=20)
    p.add_argument("--join_ratio", type=float, default=0.2)
    p.add_argument("--random_join_ratio", action="store_true")
    p.add_argument("--sampling_seed", type=int, default=2026)

    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--local_steps", type=int, default=5, help="This codebase uses local_steps as local epochs.")
    p.add_argument("--local_lr", type=float, default=0.01)
    p.add_argument("--local_momentum", type=float, default=0.9)
    p.add_argument("--local_weight_decay", type=float, default=0.0)
    p.add_argument("--grad_clip_norm", type=float, default=None)
    p.add_argument("--bn_policy", type=str, default="freeze_stats", choices=["freeze_stats", "default", "none"])

    p.add_argument("--eta", type=float, default=1.0)
    p.add_argument("--rand_percent", type=int, default=80)
    p.add_argument("--layer_idx", type=int, default=2, help="Last p parameter tensors are treated as P-subspace.")
    p.add_argument("--ala_threshold", type=float, default=0.1)
    p.add_argument("--ala_num_pre_loss", type=int, default=10)

    p.add_argument("--reset_optimizer_on_switch", action="store_true")
    p.add_argument("--reset_ala_on_switch", action="store_true")
    p.add_argument("--probe_only_window_after_switch_rounds", type=int, default=3)
    p.add_argument("--probe_clients_per_round", type=int, default=5)
    p.add_argument("--probe_steps", type=int, default=5)
    p.add_argument("--probe_batch_size", type=int, default=0, help="0 means reuse --batch_size")
    p.add_argument("--probe_fixed_seed", type=int, default=777)
    p.add_argument("--probe_fixed_batch", type=int, default=1, choices=[0, 1])
    p.add_argument("--probe_enable_top_eig", type=int, default=1, choices=[0, 1])
    p.add_argument("--probe_top_eig_power_steps", type=int, default=8)

    p.add_argument("--log_local_eval_acc", type=int, default=1, choices=[0, 1])
    p.add_argument("--local_eval_clients_per_round", type=int, default=0, help="0 means all selected clients")

    p.add_argument("--log_personalization_gain", type=int, default=1, choices=[0, 1])
    p.add_argument("--pers_clients_per_round", type=int, default=0, help="0 means all selected clients")

    p.add_argument("--theory_enable", type=int, default=1, choices=[0, 1])
    p.add_argument("--theory_lambda_enable", type=int, default=1, choices=[0, 1])
    p.add_argument("--theory_opt_drift_enable", type=int, default=1, choices=[0, 1])
    p.add_argument("--theory_only_window_after_switch_rounds", type=int, default=0,
                   help="<=0 means evaluate all rounds after task1")
    p.add_argument("--theory_clients_per_round", type=int, default=0,
                   help="client-model count for client-side theory metrics; <=0 means all selected")
    p.add_argument("--theory_global_clients_per_task", type=int, default=0,
                   help="sampled clients for global lambda_max estimate per old task; <=0 means all clients")
    p.add_argument("--theory_batch_size", type=int, default=256)
    p.add_argument("--theory_fixed_batch_seed", type=int, default=123)
    p.add_argument("--theory_lambda_power_steps", type=int, default=20)
    p.add_argument("--theory_include_personalized", type=int, default=1, choices=[0, 1],
                   help="measure client theory metrics on P_i^t in before_local phase")

    p.add_argument("--output_dir", type=str, default="./runs_mechanism_abcd")
    return p.parse_args()


def main():
    args = parse_args()
    args.task_datasets = parse_csv(args.task_datasets)
    variants = parse_experiments(args.experiments)

    if args.rounds_per_task < 1:
        raise ValueError("--rounds_per_task must be >= 1.")
    if args.times < 1:
        raise ValueError("--times must be >= 1.")
    if args.num_clients < 1:
        raise ValueError("--num_clients must be >= 1.")
    if args.local_steps < 1:
        raise ValueError("--local_steps must be >= 1.")
    if not (0 < args.join_ratio <= 1.0):
        raise ValueError("--join_ratio must be in (0, 1].")

    # data_utils.py resolves datasets via a legacy relative path ('../dataset') assuming cwd=system/.
    launch_cwd = Path.cwd()
    resolved_output_dir = Path(args.output_dir)
    if not resolved_output_dir.is_absolute():
        resolved_output_dir = (launch_cwd / resolved_output_dir).resolve()
    args.output_dir = str(resolved_output_dir)
    os.chdir(Path(__file__).resolve().parent)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA is not available, switching to CPU.")
        args.device = "cpu"

    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    suite_rows: List[Dict[str, object]] = []
    for variant in variants:
        exp_dir = output_root / variant.run_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        (exp_dir / "config_snapshot.json").write_text(
            json.dumps(config_snapshot(args, variant), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        exp_rows: List[Dict[str, object]] = []
        for run_i in range(args.times):
            run_seed = args.seed + run_i
            print(f"\n===== {variant.run_name} | run {run_i + 1}/{args.times} | seed={run_seed} =====")
            set_random_seed(run_seed)

            model = build_model(args.model_name, args.task_datasets[0], args.num_classes, args.device)
            runner_args = build_runner_args(args, model, variant)
            runner = ContinualMechanismFedALA(runner_args, variant, run_seed)
            round_rows, probe_rows, pers_rows, local_eval_rows, theory_rows, run_summary = runner.run()

            run_dir = exp_dir / f"seed_{run_seed}"
            run_dir.mkdir(parents=True, exist_ok=True)
            write_csv(run_dir / "round_metrics.csv", round_rows)
            write_csv(run_dir / "probe_metrics.csv", probe_rows)
            write_csv(run_dir / "pers_metrics.csv", pers_rows)
            write_csv(run_dir / "local_eval_metrics.csv", local_eval_rows)
            write_csv(run_dir / "theory_metrics.csv", theory_rows)
            (run_dir / "summary.json").write_text(
                json.dumps(run_summary, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            row = {
                "exp_id": variant.exp_id,
                "run_name": variant.run_name,
                "seed": run_seed,
                "freeze_mask": variant.freeze_mask,
                "personalization_enabled": int(variant.personalization_enabled),
                "probe_enabled": int(variant.probe_enabled),
                "final_acc_current_after": run_summary.get("final_acc_current_after"),
                "final_acc_old_mean_after": run_summary.get("final_acc_old_mean_after"),
                "min_delta_agg_old_mean": run_summary.get("min_delta_agg_old_mean"),
                "mean_delta_agg_old_mean_switch_window": run_summary.get(
                    "mean_delta_agg_old_mean_switch_window"
                ),
                "max_update_P_norm_mean": run_summary.get("max_update_P_norm_mean"),
                "max_update_S_norm_mean": run_summary.get("max_update_S_norm_mean"),
                "probe_rows": run_summary.get("probe_rows"),
                "pers_rows": run_summary.get("pers_rows"),
                "local_eval_rows": run_summary.get("local_eval_rows"),
                "theory_rows": run_summary.get("theory_rows"),
            }
            exp_rows.append(row)
            suite_rows.append(row)

        write_csv(exp_dir / "runs_summary.csv", exp_rows)
        (exp_dir / "runs_summary.json").write_text(
            json.dumps({"runs": exp_rows}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_csv(output_root / "suite_runs.csv", suite_rows)
    (output_root / "suite_summary.json").write_text(
        json.dumps(summarize_suite(suite_rows), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("\n===== Done =====")
    print(f"Output dir: {output_root}")
    print(f"Experiments: {[v.exp_id for v in variants]}")
    print(f"Tasks: {args.task_datasets}")


if __name__ == "__main__":
    main()

"""
Example:
python system/run_mechanism_abcd.py ^
  --task_datasets permuted-mnist-task1-npz,permuted-mnist-task2-npz,permuted-mnist-task3-npz,permuted-mnist-task4-npz,permuted-mnist-task5-npz ^
  --rounds_per_task 20 --experiments A,B,C,D --times 1 ^
  --device cuda --device_id 0 --model cnn --num_classes 10 ^
  --num_clients 20 --join_ratio 0.2 --batch_size 64 --local_steps 5 --local_lr 0.01 ^
  --eta 1.0 --rand_percent 80 --layer_idx 2 --sampling_seed 2026 ^
  --output_dir ./runs_mechanism_abcd

python system/run_mechanism_abcd.py `
  --task_datasets permuted-mnist-task1-npz,permuted-mnist-task2-npz,permuted-mnist-task3-npz,permuted-mnist-task4-npz `
  --rounds_per_task 200 `
  --experiments A,B,C,D `
  --times 1 `
  --seed 0 `
  --device cuda `
  --device_id 0 `
  --model cnn `
  --num_classes 10 `
  --num_clients 5 `
  --join_ratio 1.0 `
  --sampling_seed 2026 `
  --batch_size 10 `
  --local_steps 1 `
  --local_lr 0.005 `
  --local_momentum 0.9 `
  --local_weight_decay 0.0 `
  --bn_policy freeze_stats `
  --eta 1.0 `
  --rand_percent 80 `
  --layer_idx 2 `
  --ala_threshold 0.1 `
  --ala_num_pre_loss 10 `
  --probe_only_window_after_switch_rounds 3 `
  --probe_clients_per_round 5 `
  --probe_steps 5 `
  --probe_batch_size 64 `
  --probe_fixed_seed 777 `
  --probe_fixed_batch 1 `
  --probe_enable_top_eig 1 `
  --probe_top_eig_power_steps 8 `
  --log_personalization_gain 1 `
  --pers_clients_per_round 0 `
  --theory_enable 1 `
  --theory_lambda_enable 1 `
  --theory_opt_drift_enable 1 `
  --theory_only_window_after_switch_rounds 0 `
  --theory_clients_per_round 0 `
  --theory_global_clients_per_task 0 `
  --theory_batch_size 256 `
  --theory_fixed_batch_seed 123 `
  --theory_lambda_power_steps 20 `
  --theory_include_personalized 1 `
  --output_dir ./runs_mechanism_abcd_permuted_mnist

"""
