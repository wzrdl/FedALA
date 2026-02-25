import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EXP_STYLE = {
    "A": {"label": "A (Full-FedALA)", "color": "#d62728", "marker": "o"},
    "B": {"label": "B (No-Pers)", "color": "#1f77b4", "marker": "s"},
    "C": {"label": "C (Freeze-P)", "color": "#2ca02c", "marker": "^"},
    "D": {"label": "D (Freeze-S)", "color": "#ff7f0e", "marker": "D"},
}


def parse_args():
    p = argparse.ArgumentParser(description="Plot A/B/C/D mechanism experiment results.")
    p.add_argument("--runs_root", required=True)
    p.add_argument("--output_dir", default=None)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title_suffix", default="")
    p.add_argument("--shock_window_rounds", type=int, default=10)
    p.add_argument("--scatter_window_rounds", type=int, default=5)
    p.add_argument("--progress_bins", type=int, default=30)
    p.add_argument("--theory_phase", default="before_local", choices=["before_local", "after_local", "after_agg"])
    p.add_argument("--theory_scope", default="global", choices=["global", "client"])
    p.add_argument("--probe_experiment", default="A")
    p.add_argument("--rolling", type=int, default=1)
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _seed_from_name(name: str):
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("seed_", 1)[1])
    except ValueError:
        return None


def _exp_code(name: str):
    return name.split("_", 1)[0]


def load_runs(runs_root: Path):
    round_parts, probe_parts, theory_parts = [], [], []
    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_code = _exp_code(exp_dir.name)
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = _seed_from_name(seed_dir.name)
            if seed is None:
                continue
            round_df = safe_read_csv(seed_dir / "round_metrics.csv")
            probe_df = safe_read_csv(seed_dir / "probe_metrics.csv")
            theory_df = safe_read_csv(seed_dir / "theory_metrics.csv")
            for name, df in [("round", round_df), ("probe", probe_df), ("theory", theory_df)]:
                if df.empty:
                    continue
                df["exp_dir"] = exp_dir.name
                df["exp_code"] = exp_code
                df["exp_label"] = EXP_STYLE.get(exp_code, {}).get("label", exp_code)
                df["seed"] = seed
                if name == "round":
                    round_parts.append(df)
                elif name == "probe":
                    probe_parts.append(df)
                else:
                    theory_parts.append(df)
    if not round_parts:
        raise FileNotFoundError(f"No round_metrics.csv found under {runs_root}")
    return {
        "round": pd.concat(round_parts, ignore_index=True, sort=False),
        "probe": pd.concat(probe_parts, ignore_index=True, sort=False) if probe_parts else pd.DataFrame(),
        "theory": pd.concat(theory_parts, ignore_index=True, sort=False) if theory_parts else pd.DataFrame(),
    }


def title_with_suffix(base: str, suffix: str):
    s = (suffix or "").strip()
    return f"{base} ({s})" if s else base


def save_fig(fig, path: Path, dpi: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def maybe_rolling(s: pd.Series, window: int):
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def infer_switch_info(round_df: pd.DataFrame):
    if not {"global_round", "task_idx", "is_task_switch_round"}.issubset(round_df.columns):
        return []
    d = round_df.loc[
        pd.to_numeric(round_df["is_task_switch_round"], errors="coerce").fillna(0).astype(int) == 1,
        ["global_round", "task_idx"],
    ].copy()
    if d.empty:
        return []
    d["global_round"] = pd.to_numeric(d["global_round"], errors="coerce")
    d["task_idx"] = pd.to_numeric(d["task_idx"], errors="coerce")
    d = d.dropna().drop_duplicates().sort_values(["global_round", "task_idx"])
    return [(int(r.global_round), int(r.task_idx)) for _, r in d.iterrows()]


def add_switch_lines(ax, switch_info, annotate=False):
    if not switch_info:
        return
    y0, y1 = ax.get_ylim()
    dy = (y1 - y0) if y1 > y0 else 1.0
    for g_round, task_idx in switch_info:
        ax.axvline(g_round, color="0.6", linestyle="--", linewidth=1, alpha=0.8)
        if annotate:
            ax.text(g_round + 1, y1 - 0.05 * dy, f"Task {task_idx}", rotation=90, va="top", ha="left", fontsize=8, color="0.35")


def aggregate_round(round_df: pd.DataFrame, value_cols):
    keep = ["exp_code", "exp_label", "global_round"]
    for c in ["task_idx", "round_in_task"]:
        if c in round_df.columns:
            keep.append(c)
    keep += [c for c in value_cols if c in round_df.columns]
    d = round_df[keep].copy()
    for c in d.columns:
        if c not in ["exp_code", "exp_label"]:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    agg_spec = {c: "mean" for c in d.columns if c not in ["exp_code", "exp_label", "global_round"]}
    out = d.groupby(["exp_code", "exp_label", "global_round"], dropna=False).agg(agg_spec).reset_index()
    return out.sort_values(["exp_code", "global_round"]).reset_index(drop=True)


def plot_fig1(round_df, out_dir: Path, dpi: int, suffix: str, rolling: int):
    if "acc_old_mean_after" not in round_df.columns:
        return None
    d = round_df[round_df["exp_code"].isin(["A", "B"])].copy()
    if d.empty:
        return None
    agg = aggregate_round(d, ["acc_old_mean_after"])
    switch_info = infer_switch_info(d)
    fig, ax = plt.subplots(figsize=(10.6, 4.2))
    for code in ["A", "B"]:
        part = agg[agg["exp_code"] == code]
        if part.empty:
            continue
        style = EXP_STYLE.get(code, {})
        ax.plot(part["global_round"], maybe_rolling(part["acc_old_mean_after"], rolling), color=style.get("color"), linewidth=2, label=style.get("label", code))
    ax.set_title(title_with_suffix("Fig1 Macro Forgetting Trajectory", suffix))
    ax.set_xlabel("Global Communication Round")
    ax.set_ylabel("Avg Accuracy on Old Tasks")
    ax.grid(True, linestyle=":", alpha=0.35)
    if ax.has_data():
        add_switch_lines(ax, switch_info, annotate=True)
        ax.legend(frameon=False)
    out = out_dir / "fig1_macro_forgetting_trajectory.png"
    save_fig(fig, out, dpi)
    return out.name


def plot_fig2(round_df, out_dir: Path, dpi: int, suffix: str, shock_window_rounds: int):
    need = {"delta_agg_old_mean", "task_idx", "round_in_task"}
    if not need.issubset(round_df.columns):
        return None
    d = round_df[round_df["exp_code"].isin(["A", "B"])].copy()
    for c in ["delta_agg_old_mean", "task_idx", "round_in_task"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[(d["task_idx"] > 1) & (d["round_in_task"] >= 1) & (d["round_in_task"] <= shock_window_rounds)]
    if d.empty:
        return None
    d["rel_round"] = d["round_in_task"] - 1
    agg = (
        d.groupby(["exp_code", "rel_round"])["delta_agg_old_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "y_mean", "std": "y_std"})
    )
    fig, ax = plt.subplots(figsize=(8.6, 4.2))
    for code in ["A", "B"]:
        part = agg[agg["exp_code"] == code]
        if part.empty:
            continue
        style = EXP_STYLE.get(code, {})
        x = part["rel_round"]
        y = part["y_mean"]
        ax.plot(x, y, color=style.get("color"), marker=style.get("marker"), linewidth=2, label=style.get("label", code))
        if (part["count"] > 1).any():
            err = part["y_std"].fillna(0)
            ax.fill_between(x, y - err, y + err, color=style.get("color"), alpha=0.12)
    ax.axhline(0, color="black", linewidth=1, alpha=0.8)
    ax.set_title(title_with_suffix("Fig2 Aggregation Shock Dynamics", suffix))
    ax.set_xlabel("Rounds After Task Switch")
    ax.set_ylabel("Aggregation Shock (ΔAgg old-task)")
    ax.grid(True, linestyle=":", alpha=0.35)
    if ax.has_data():
        ax.legend(frameon=False)
    out = out_dir / "fig2_aggregation_shock_dynamics.png"
    save_fig(fig, out, dpi)
    return out.name


def plot_fig3(round_df, out_dir: Path, dpi: int, suffix: str, rolling: int):
    need = {"update_P_norm_mean", "update_S_norm_mean"}
    if not need.issubset(round_df.columns):
        return None
    d = round_df[round_df["exp_code"].isin(["A", "B"])].copy()
    if d.empty:
        return None
    agg = aggregate_round(d, ["update_P_norm_mean", "update_S_norm_mean"])
    switch_info = infer_switch_info(d)
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 6.5), sharex=True)
    for ax, col, ylabel in [
        (axes[0], "update_P_norm_mean", "P-subspace Update Norm"),
        (axes[1], "update_S_norm_mean", "S-subspace Update Norm"),
    ]:
        for code in ["A", "B"]:
            part = agg[agg["exp_code"] == code]
            if part.empty:
                continue
            style = EXP_STYLE.get(code, {})
            ax.plot(part["global_round"], maybe_rolling(part[col], rolling), color=style.get("color"), linewidth=1.9, label=style.get("label", code))
        ax.grid(True, linestyle=":", alpha=0.35)
        ax.set_ylabel(ylabel)
        if ax.has_data():
            add_switch_lines(ax, switch_info, annotate=False)
    axes[0].set_title(title_with_suffix("Fig3 Subspace Update Norms (A vs B)", suffix))
    axes[1].set_xlabel("Global Communication Round")
    if axes[0].has_data():
        axes[0].legend(frameon=False, ncol=2)
    out = out_dir / "fig3_subspace_update_norms.png"
    save_fig(fig, out, dpi)
    return out.name


def plot_fig4(round_df, out_dir: Path, dpi: int, suffix: str, scatter_window_rounds: int):
    need = {"update_P_norm_mean", "delta_agg_old_mean", "task_idx", "round_in_task"}
    if not need.issubset(round_df.columns):
        return None
    d = round_df.copy()
    for c in ["update_P_norm_mean", "delta_agg_old_mean", "task_idx", "round_in_task"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["exp_code"].isin(["A", "B", "C", "D"])]
    d = d[(d["task_idx"] > 1) & (d["round_in_task"] >= 1) & (d["round_in_task"] <= scatter_window_rounds)]
    d = d.dropna(subset=["update_P_norm_mean", "delta_agg_old_mean"])
    if d.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 5.3))
    for code in ["A", "B", "C", "D"]:
        part = d[d["exp_code"] == code]
        if part.empty:
            continue
        style = EXP_STYLE.get(code, {})
        ax.scatter(part["update_P_norm_mean"], part["delta_agg_old_mean"], s=36, alpha=0.75, c=style.get("color"), marker=style.get("marker"), label=style.get("label", code), edgecolors="none")
    x = d["update_P_norm_mean"].to_numpy(dtype=float)
    y = d["delta_agg_old_mean"].to_numpy(dtype=float)
    if len(x) >= 2:
        slope, intercept = np.polyfit(x, y, 1)
        xline = np.linspace(np.nanmin(x), np.nanmax(x), 100)
        yline = slope * xline + intercept
        ax.plot(xline, yline, color="black", linestyle="--", linewidth=2, label="Overall fit")
        yhat = slope * x + intercept
        ss_res = float(np.sum((y - yhat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = np.nan if ss_tot <= 0 else 1.0 - ss_res / ss_tot
        ax.text(0.98, 0.03, f"slope={slope:.3f}\nR²={r2:.3f}" if np.isfinite(r2) else f"slope={slope:.3f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=9, bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="0.8"))
    ax.axhline(0, color="0.25", linewidth=1, alpha=0.6)
    ax.set_title(title_with_suffix("Fig4 Causal Scatter: U_P vs ΔAgg", suffix))
    ax.set_xlabel("P-subspace Update Norm (U_P)")
    ax.set_ylabel("Aggregation Shock (ΔAgg on Old Tasks)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=9)
    out = out_dir / "fig4_causal_scatter_pnorm_vs_shock.png"
    save_fig(fig, out, dpi)
    return out.name


def _progress_binned_curve(d: pd.DataFrame, bins: int):
    d = d[["exp_code", "acc_current_after", "acc_old_mean_after"]].dropna().copy()
    if d.empty:
        return pd.DataFrame()
    xmin = float(d["acc_current_after"].min())
    xmax = float(d["acc_current_after"].max())
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        return pd.DataFrame()
    edges = np.linspace(xmin, xmax, bins + 1)
    d["xbin"] = pd.cut(d["acc_current_after"], bins=edges, include_lowest=True, duplicates="drop")
    out = d.groupby(["exp_code", "xbin"], observed=True)["acc_old_mean_after"].agg(["mean", "std", "count"]).reset_index()
    out["x_mid"] = [(iv.left + iv.right) / 2 for iv in out["xbin"]]
    return out.sort_values(["exp_code", "x_mid"]).reset_index(drop=True)


def plot_fig5(round_df, out_dir: Path, dpi: int, suffix: str, progress_bins: int):
    need = {"acc_current_after", "acc_old_mean_after", "task_idx"}
    if not need.issubset(round_df.columns):
        return None
    d = round_df[round_df["exp_code"].isin(["A", "C", "D"])].copy()
    for c in ["acc_current_after", "acc_old_mean_after", "task_idx"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d[d["task_idx"] > 1].dropna(subset=["acc_current_after", "acc_old_mean_after"])
    if d.empty:
        return None
    binned = _progress_binned_curve(d, progress_bins)
    if binned.empty:
        return None
    fig, ax = plt.subplots(figsize=(8.8, 5.5))
    for code in ["A", "C", "D"]:
        raw = d[d["exp_code"] == code]
        if not raw.empty:
            ax.scatter(raw["acc_current_after"], raw["acc_old_mean_after"], s=10, alpha=0.12, c=EXP_STYLE.get(code, {}).get("color"), edgecolors="none")
        part = binned[binned["exp_code"] == code]
        if part.empty:
            continue
        style = EXP_STYLE.get(code, {})
        ax.plot(part["x_mid"], part["mean"], color=style.get("color"), marker=style.get("marker"), linewidth=2, markersize=4, label=style.get("label", code))
    ax.set_title(title_with_suffix("Fig5 Progress-Matched Anti-Confounder", suffix))
    ax.set_xlabel("Accuracy on Current Task")
    ax.set_ylabel("Avg Accuracy on Old Tasks")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    out = out_dir / "fig5_progress_matched_anti_confounder.png"
    save_fig(fig, out, dpi)
    return out.name


def aggregate_theory(theory_df: pd.DataFrame, phase: str, scope: str):
    need = {"phase", "scope", "global_round", "exp_code", "lambda_max", "opt_drift"}
    if theory_df.empty or not need.issubset(theory_df.columns):
        return pd.DataFrame()
    d = theory_df[(theory_df["phase"] == phase) & (theory_df["scope"] == scope)].copy()
    if d.empty:
        return d
    for c in ["global_round", "lambda_max", "opt_drift"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    d = d.dropna(subset=["global_round"])
    return d.groupby(["exp_code", "global_round"])[["lambda_max", "opt_drift"]].mean().reset_index().sort_values(["exp_code", "global_round"])


def plot_fig6(round_df, theory_df, out_dir: Path, dpi: int, suffix: str, phase: str, scope: str, rolling: int):
    if theory_df.empty:
        return None
    d = theory_df[theory_df["exp_code"].isin(["A", "B"])].copy()
    agg = aggregate_theory(d, phase=phase, scope=scope)
    if agg.empty:
        return None
    switch_info = infer_switch_info(round_df[round_df["exp_code"].isin(["A", "B"])])
    fig, ax1 = plt.subplots(figsize=(10.8, 4.8))
    ax2 = ax1.twinx()
    handles, labels = [], []
    for code in ["A", "B"]:
        part = agg[agg["exp_code"] == code]
        if part.empty:
            continue
        style = EXP_STYLE.get(code, {})
        h1, = ax1.plot(part["global_round"], maybe_rolling(part["lambda_max"], rolling), color=style.get("color"), linestyle="-", linewidth=2)
        h2, = ax2.plot(part["global_round"], maybe_rolling(part["opt_drift"], rolling), color=style.get("color"), linestyle="--", linewidth=1.8)
        handles += [h1, h2]
        labels += [f"{style.get('label', code)} λ_max", f"{style.get('label', code)} opt_drift"]
    ax1.set_title(title_with_suffix(f"Fig6 Theory Metrics ({phase}/{scope})", suffix))
    ax1.set_xlabel("Global Communication Round")
    ax1.set_ylabel("Hessian Top Eigenvalue (λ_max)")
    ax2.set_ylabel("Optimal Drift")
    ax1.grid(True, linestyle=":", alpha=0.35)
    if ax1.has_data():
        add_switch_lines(ax1, switch_info, annotate=False)
        ax1.legend(handles, labels, frameon=False, fontsize=8, ncol=2, loc="upper right")
    out = out_dir / f"fig6_theory_metrics_{phase}_{scope}.png"
    save_fig(fig, out, dpi)
    return out.name


def _probe_pick_exp(probe_df: pd.DataFrame, preferred: str):
    if probe_df.empty or "exp_code" not in probe_df.columns:
        return None
    ok = probe_df.copy()
    if "probe_status" in ok.columns:
        ok = ok[ok["probe_status"] == "ok"]
    if ok.empty:
        return None
    codes = list(pd.unique(ok["exp_code"]))
    if preferred in codes:
        return preferred
    for c in ["A", "B", "C", "D"]:
        if c in codes:
            return c
    return codes[0]


def _probe_start_summary(d: pd.DataFrame):
    pairs = [
        ("U_norm", "G_U_norm", "P_U_norm"),
        ("U_P_norm", "G_U_P_norm", "P_U_P_norm"),
        ("U_S_norm", "G_U_S_norm", "P_U_S_norm"),
        ("U_P_ratio", "G_U_P_ratio", "P_U_P_ratio"),
        ("Loss_after", "G_loss_after", "P_loss_after"),
        ("TopEig_proj_ratio", "G_top_eig_proj_ratio", "P_top_eig_proj_ratio"),
    ]
    rows = []
    for name, gcol, pcol in pairs:
        if gcol not in d.columns or pcol not in d.columns:
            continue
        g = pd.to_numeric(d[gcol], errors="coerce").dropna()
        p = pd.to_numeric(d[pcol], errors="coerce").dropna()
        if g.empty and p.empty:
            continue
        rows.append({"metric": name, "G_start": float(g.mean()) if not g.empty else np.nan, "P_start": float(p.mean()) if not p.empty else np.nan})
    return pd.DataFrame(rows)


def _probe_pair_summary(d: pd.DataFrame):
    cols = [
        "grad_alignment",
        "update_alignment",
        "risk_gap_loss_after_P_minus_G",
        "risk_gap_loss_delta_P_minus_G",
        "top_eig_proj_gap_P_minus_G",
    ]
    rows = []
    for c in cols:
        if c not in d.columns:
            continue
        x = pd.to_numeric(d[c], errors="coerce").dropna()
        if x.empty:
            continue
        rows.append({"metric": c, "value": float(x.mean())})
    return pd.DataFrame(rows)


def plot_fig7(probe_df, out_dir: Path, dpi: int, suffix: str, preferred_exp: str):
    if probe_df.empty:
        return None
    d = probe_df.copy()
    if "probe_status" in d.columns:
        d = d[d["probe_status"] == "ok"]
    if d.empty:
        return None
    exp_code = _probe_pick_exp(d, preferred_exp)
    if exp_code is None:
        return None
    d = d[d["exp_code"] == exp_code].copy()
    start_s = _probe_start_summary(d)
    pair_s = _probe_pair_summary(d)
    if start_s.empty and pair_s.empty:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 4.8), gridspec_kw={"width_ratios": [1.8, 1.0]})
    ax1, ax2 = axes
    if not start_s.empty:
        x = np.arange(len(start_s))
        w = 0.38
        ax1.bar(x - w / 2, start_s["G_start"], width=w, color="#4c78a8", label="G-start")
        ax1.bar(x + w / 2, start_s["P_start"], width=w, color="#e45756", label="P-start")
        ax1.set_xticks(x)
        ax1.set_xticklabels(start_s["metric"], rotation=25, ha="right")
        ax1.set_ylabel("Mean Value")
        ax1.set_title(f"Probe start-state metrics ({EXP_STYLE.get(exp_code, {}).get('label', exp_code)})")
        ax1.grid(True, axis="y", linestyle=":", alpha=0.35)
        ax1.legend(frameon=False)
    else:
        ax1.axis("off")
    if not pair_s.empty:
        x2 = np.arange(len(pair_s))
        colors = ["#72b7b2" if v >= 0 else "#f28e2b" for v in pair_s["value"]]
        ax2.bar(x2, pair_s["value"], color=colors)
        ax2.axhline(0, color="0.25", linewidth=1)
        ax2.set_xticks(x2)
        ax2.set_xticklabels(pair_s["metric"], rotation=25, ha="right")
        ax2.set_ylabel("Mean Value")
        ax2.set_title("Pairwise probe diagnostics")
        ax2.grid(True, axis="y", linestyle=":", alpha=0.35)
    else:
        ax2.axis("off")
    fig.suptitle(title_with_suffix(f"Fig7 Microscopic Probe P1 (n={len(d)})", suffix), y=1.02)
    out = out_dir / "fig7_microscopic_probe_p1.png"
    save_fig(fig, out, dpi)
    return out.name


def export_aggregates(tables, out_dir: Path, phase: str, scope: str):
    exported = {}
    agg_round = aggregate_round(
        tables["round"],
        [
            "acc_current_after",
            "acc_old_mean_after",
            "delta_agg_old_mean",
            "update_norm_mean",
            "update_P_norm_mean",
            "update_S_norm_mean",
            "update_P_ratio_mean",
            "probe_grad_alignment_mean",
            "probe_update_alignment_mean",
        ],
    )
    if not agg_round.empty:
        p = out_dir / "aggregated_round_metrics_mean.csv"
        agg_round.to_csv(p, index=False)
        exported["aggregated_round_metrics_mean"] = p.name
    agg_theory = aggregate_theory(tables["theory"], phase, scope) if not tables["theory"].empty else pd.DataFrame()
    if not agg_theory.empty:
        p = out_dir / f"aggregated_theory_metrics_{phase}_{scope}.csv"
        agg_theory.to_csv(p, index=False)
        exported["aggregated_theory_metrics"] = p.name
    if not tables["probe"].empty:
        rows = []
        d = tables["probe"].copy()
        if "probe_status" in d.columns:
            d = d[d["probe_status"] == "ok"]
        for code, part in d.groupby("exp_code"):
            s1 = _probe_start_summary(part)
            if not s1.empty:
                s1.insert(0, "summary_type", "start_state")
                s1.insert(0, "exp_code", code)
                rows.append(s1)
            s2 = _probe_pair_summary(part)
            if not s2.empty:
                s2.insert(0, "summary_type", "pairwise")
                s2.insert(0, "exp_code", code)
                rows.append(s2)
        if rows:
            p = out_dir / "aggregated_probe_summary.csv"
            pd.concat(rows, ignore_index=True, sort=False).to_csv(p, index=False)
            exported["aggregated_probe_summary"] = p.name
    return exported


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")
    out_dir = Path(args.output_dir) if args.output_dir else (runs_root / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    tables = load_runs(runs_root)

    # common numeric coercion (important for NaN-safe plotting)
    for col in [
        "global_round",
        "task_idx",
        "round_in_task",
        "acc_current_after",
        "acc_old_mean_after",
        "delta_agg_old_mean",
        "update_norm_mean",
        "update_P_norm_mean",
        "update_S_norm_mean",
    ]:
        if col in tables["round"].columns:
            tables["round"][col] = pd.to_numeric(tables["round"][col], errors="coerce")

    generated = []
    for fn in [
        lambda: plot_fig1(tables["round"], out_dir, args.dpi, args.title_suffix, args.rolling),
        lambda: plot_fig2(tables["round"], out_dir, args.dpi, args.title_suffix, args.shock_window_rounds),
        lambda: plot_fig3(tables["round"], out_dir, args.dpi, args.title_suffix, args.rolling),
        lambda: plot_fig4(tables["round"], out_dir, args.dpi, args.title_suffix, args.scatter_window_rounds),
        lambda: plot_fig5(tables["round"], out_dir, args.dpi, args.title_suffix, args.progress_bins),
        lambda: plot_fig6(tables["round"], tables["theory"], out_dir, args.dpi, args.title_suffix, args.theory_phase, args.theory_scope, args.rolling),
        lambda: plot_fig7(tables["probe"], out_dir, args.dpi, args.title_suffix, args.probe_experiment),
    ]:
        try:
            name = fn()
            if name:
                generated.append(name)
        except Exception as e:
            print(f"[warn] plotting step failed: {e}")

    exports = export_aggregates(tables, out_dir, args.theory_phase, args.theory_scope)
    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(out_dir),
        "figures_generated": generated,
        "exports": exports,
        "config": {
            "dpi": args.dpi,
            "title_suffix": args.title_suffix,
            "shock_window_rounds": args.shock_window_rounds,
            "scatter_window_rounds": args.scatter_window_rounds,
            "progress_bins": args.progress_bins,
            "theory_phase": args.theory_phase,
            "theory_scope": args.theory_scope,
            "probe_experiment": args.probe_experiment,
            "rolling": args.rolling,
        },
        "data_counts": {
            "round_rows": int(len(tables["round"])),
            "probe_rows": int(len(tables["probe"])),
            "theory_rows": int(len(tables["theory"])),
            "experiments": sorted(map(str, pd.unique(tables["round"]["exp_code"]))),
            "seeds": sorted(map(int, pd.unique(tables["round"]["seed"]))),
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(generated)} figures to {out_dir}")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
