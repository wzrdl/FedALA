import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


TASK_COLORS = [
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#7f7f7f",
    "#bcbd22",
    "#17becf",
]


def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Plot local mean accuracy curves (from pers_metrics.csv) for each experiment. "
            "Curves are personalized local accuracy on old tasks."
        )
    )
    p.add_argument("--runs_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title_suffix", type=str, default="")
    p.add_argument("--rolling", type=int, default=1)
    p.add_argument(
        "--include_mean",
        action="store_true",
        help="Overlay mean personalized local accuracy across old tasks.",
    )
    p.add_argument(
        "--client_weighted",
        action="store_true",
        help="Use test_samples-weighted mean across clients for each (round, old_task).",
    )
    p.add_argument(
        "--also_plot_global_local_baseline",
        action="store_true",
        help="Overlay global-model local accuracy (from global_acc_old_task) as dashed curves.",
    )
    return p.parse_args()


def safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parse_seed_dir(name: str) -> Optional[int]:
    if not name.startswith("seed_"):
        return None
    try:
        return int(name.split("seed_", 1)[1])
    except ValueError:
        return None


def parse_exp_code(exp_dir_name: str) -> str:
    return exp_dir_name.split("_", 1)[0]


def title_with_suffix(base: str, suffix: str) -> str:
    suffix = (suffix or "").strip()
    return f"{base} ({suffix})" if suffix else base


def maybe_rolling(s: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def load_pers_long(runs_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    pers_parts: List[pd.DataFrame] = []
    switch_rows: List[Dict[str, object]] = []

    for exp_dir in sorted(runs_root.iterdir()):
        if not exp_dir.is_dir():
            continue
        exp_code = parse_exp_code(exp_dir.name)
        for seed_dir in sorted(exp_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            seed = parse_seed_dir(seed_dir.name)
            if seed is None:
                continue

            pers_df = safe_read_csv(seed_dir / "pers_metrics.csv")
            if not pers_df.empty:
                pers_df["exp_dir"] = exp_dir.name
                pers_df["exp_code"] = exp_code
                pers_df["seed"] = seed
                pers_parts.append(pers_df)

            round_df = safe_read_csv(seed_dir / "round_metrics.csv")
            if round_df.empty:
                continue
            need = {"global_round", "task_idx", "is_task_switch_round"}
            if not need.issubset(round_df.columns):
                continue
            tmp = round_df[list(need)].copy()
            for c in ["global_round", "task_idx", "is_task_switch_round"]:
                tmp[c] = pd.to_numeric(tmp[c], errors="coerce")
            tmp = tmp[tmp["is_task_switch_round"].fillna(0).astype(int) == 1]
            for _, r in tmp.dropna(subset=["global_round", "task_idx"]).drop_duplicates().iterrows():
                switch_rows.append(
                    {
                        "exp_dir": exp_dir.name,
                        "exp_code": exp_code,
                        "seed": seed,
                        "global_round": int(r["global_round"]),
                        "task_idx": int(r["task_idx"]),
                    }
                )

    if not pers_parts:
        raise FileNotFoundError(f"No pers_metrics.csv found under {runs_root}")
    pers_df = pd.concat(pers_parts, ignore_index=True, sort=False)
    switch_df = pd.DataFrame(switch_rows) if switch_rows else pd.DataFrame(
        columns=["exp_dir", "exp_code", "seed", "global_round", "task_idx"]
    )
    return pers_df, switch_df


def _weighted_mean(values: pd.Series, weights: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce")
    w = pd.to_numeric(weights, errors="coerce")
    mask = v.notna() & w.notna() & (w > 0)
    if not mask.any():
        return float("nan")
    return float(np.average(v[mask].to_numpy(dtype=float), weights=w[mask].to_numpy(dtype=float)))


def _aggregate_clients_per_seed(
    pers_df: pd.DataFrame,
    acc_col: str,
    *,
    client_weighted: bool,
) -> pd.DataFrame:
    need = {"exp_dir", "exp_code", "seed", "global_round", "old_task_idx", acc_col}
    if not need.issubset(pers_df.columns):
        return pd.DataFrame()
    d = pers_df.copy()
    for c in ["global_round", "task_idx", "round_in_task", "old_task_idx", "test_samples", acc_col]:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")
    # Exclude summary rows old_task_idx=0 (OLD_TASKS_MEAN) because they do not contain acc columns.
    d = d[d["old_task_idx"] > 0]
    d = d.dropna(subset=["global_round", "old_task_idx", acc_col])
    if d.empty:
        return pd.DataFrame()

    group_cols = ["exp_dir", "exp_code", "seed", "global_round", "old_task_idx"]
    if client_weighted and "test_samples" in d.columns:
        rows = []
        for key, part in d.groupby(group_cols, dropna=False):
            mean_val = _weighted_mean(part[acc_col], part["test_samples"])
            rows.append(
                {
                    "exp_dir": key[0],
                    "exp_code": key[1],
                    "seed": key[2],
                    "global_round": key[3],
                    "old_task_idx": key[4],
                    "acc_seed_mean": mean_val,
                    "num_client_rows": int(len(part)),
                    "sum_test_samples": float(pd.to_numeric(part["test_samples"], errors="coerce").fillna(0).sum()),
                }
            )
        out = pd.DataFrame(rows)
    else:
        out = (
            d.groupby(group_cols, dropna=False)[acc_col]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "acc_seed_mean", "count": "num_client_rows"})
        )
        out["sum_test_samples"] = np.nan

    # Mean across old tasks per round (per seed)
    mean_rows = (
        out.groupby(["exp_dir", "exp_code", "seed", "global_round"], dropna=False)["acc_seed_mean"]
        .mean()
        .reset_index()
        .assign(old_task_idx=0)
    )
    mean_rows["num_client_rows"] = np.nan
    mean_rows["sum_test_samples"] = np.nan

    out = pd.concat([out, mean_rows], ignore_index=True, sort=False)
    return out


def aggregate_across_seeds(seed_level_df: pd.DataFrame) -> pd.DataFrame:
    if seed_level_df.empty:
        return seed_level_df
    d = seed_level_df.copy()
    for c in ["global_round", "old_task_idx", "acc_seed_mean"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = (
        d.groupby(["exp_dir", "exp_code", "global_round", "old_task_idx"], dropna=False)["acc_seed_mean"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std", "count": "num_seeds"})
        .sort_values(["exp_dir", "old_task_idx", "global_round"])
    )
    return agg


def aggregate_switches(switch_df: pd.DataFrame) -> pd.DataFrame:
    if switch_df.empty:
        return switch_df
    d = switch_df.copy()
    for c in ["global_round", "task_idx"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    return d.groupby(["exp_dir", "exp_code", "global_round", "task_idx"], dropna=False).size().reset_index(name="n")


def save_fig(fig: plt.Figure, out_path: Path, dpi: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_one_experiment(
    exp_dir: str,
    exp_code: str,
    personal_agg: pd.DataFrame,
    switch_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    title_suffix: str,
    include_mean: bool,
    rolling: int,
    global_agg: Optional[pd.DataFrame] = None,
) -> str:
    part_p = personal_agg[personal_agg["exp_dir"] == exp_dir].copy()
    if part_p.empty:
        raise ValueError(f"No personalized local accuracy data for {exp_dir}")

    fig, ax = plt.subplots(figsize=(11.4, 5.4))

    task_ids = sorted(int(x) for x in pd.unique(part_p["old_task_idx"]) if int(x) > 0)
    for idx, t in enumerate(task_ids):
        tdf = part_p[part_p["old_task_idx"] == t].sort_values("global_round").copy()
        if tdf.empty:
            continue
        color = TASK_COLORS[idx % len(TASK_COLORS)]
        y = maybe_rolling(tdf["acc_mean"], rolling)
        ax.plot(tdf["global_round"], y, color=color, linewidth=2, label=f"Local-Pers Task {t}")
        if (tdf["num_seeds"] > 1).any():
            err = tdf["acc_std"].fillna(0)
            ax.fill_between(tdf["global_round"], y - err, y + err, color=color, alpha=0.12)

    if include_mean:
        mdf = part_p[part_p["old_task_idx"] == 0].sort_values("global_round").copy()
        if not mdf.empty:
            ax.plot(
                mdf["global_round"],
                maybe_rolling(mdf["acc_mean"], rolling),
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean(old tasks, local-pers)",
            )

    if global_agg is not None and not global_agg.empty:
        part_g = global_agg[global_agg["exp_dir"] == exp_dir].copy()
        for idx, t in enumerate(task_ids):
            tdf = part_g[part_g["old_task_idx"] == t].sort_values("global_round").copy()
            if tdf.empty:
                continue
            color = TASK_COLORS[idx % len(TASK_COLORS)]
            ax.plot(
                tdf["global_round"],
                maybe_rolling(tdf["acc_mean"], rolling),
                color=color,
                linewidth=1.4,
                linestyle=":",
                alpha=0.9,
                label=f"Local-Global Task {t}" if idx == 0 else None,  # avoid huge legend duplication
            )
        if include_mean:
            mdf = part_g[part_g["old_task_idx"] == 0].sort_values("global_round").copy()
            if not mdf.empty:
                ax.plot(
                    mdf["global_round"],
                    maybe_rolling(mdf["acc_mean"], rolling),
                    color="0.25",
                    linewidth=1.8,
                    linestyle="-.",
                    label="Mean(old tasks, local-global)",
                )

    s_part = switch_df[switch_df["exp_dir"] == exp_dir] if not switch_df.empty else pd.DataFrame()
    if not s_part.empty:
        y0, y1 = ax.get_ylim()
        dy = (y1 - y0) if y1 > y0 else 1.0
        for _, r in s_part.sort_values("global_round").iterrows():
            g_round = int(r["global_round"])
            t_idx = int(r["task_idx"])
            ax.axvline(g_round, color="0.6", linestyle="--", linewidth=1, alpha=0.75)
            ax.text(g_round + 1, y1 - 0.05 * dy, f"Task {t_idx}", rotation=90, fontsize=8, color="0.35", va="top")

    ax.set_title(title_with_suffix(f"{exp_dir}: Local Mean Accuracy on Old Tasks", title_suffix))
    ax.set_xlabel("Global Communication Round")
    ax.set_ylabel("Accuracy (client-mean on old tasks)")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, fontsize=9, ncol=2)

    out_name = f"{exp_dir}_local_mean_acc_old_tasks.png"
    save_fig(fig, out_dir / out_name, dpi)
    return out_name


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")
    out_dir = Path(args.output_dir) if args.output_dir else (runs_root / "figures_local_mean_acc")
    out_dir.mkdir(parents=True, exist_ok=True)

    pers_df, switch_df = load_pers_long(runs_root)
    seed_personal = _aggregate_clients_per_seed(
        pers_df,
        acc_col="personalized_acc_old_task",
        client_weighted=bool(args.client_weighted),
    )
    agg_personal = aggregate_across_seeds(seed_personal)
    if agg_personal.empty:
        raise RuntimeError("No personalized_acc_old_task data available after aggregation.")

    agg_global = pd.DataFrame()
    if args.also_plot_global_local_baseline and "global_acc_old_task" in pers_df.columns:
        seed_global = _aggregate_clients_per_seed(
            pers_df,
            acc_col="global_acc_old_task",
            client_weighted=bool(args.client_weighted),
        )
        agg_global = aggregate_across_seeds(seed_global)

    sw_agg = aggregate_switches(switch_df)

    generated = []
    for exp_dir, exp_code in (
        agg_personal[["exp_dir", "exp_code"]].drop_duplicates().sort_values(["exp_dir"]).itertuples(index=False, name=None)
    ):
        name = plot_one_experiment(
            exp_dir=exp_dir,
            exp_code=exp_code,
            personal_agg=agg_personal,
            switch_df=sw_agg,
            out_dir=out_dir,
            dpi=args.dpi,
            title_suffix=args.title_suffix,
            include_mean=bool(args.include_mean),
            rolling=int(args.rolling),
            global_agg=agg_global if not agg_global.empty else None,
        )
        generated.append(name)

    agg_personal.to_csv(out_dir / "aggregated_local_personalized_acc_old_tasks.csv", index=False)
    if not agg_global.empty:
        agg_global.to_csv(out_dir / "aggregated_local_global_acc_old_tasks.csv", index=False)

    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(out_dir),
        "figures_generated": generated,
        "config": {
            "dpi": args.dpi,
            "title_suffix": args.title_suffix,
            "rolling": int(args.rolling),
            "include_mean": bool(args.include_mean),
            "client_weighted": bool(args.client_weighted),
            "also_plot_global_local_baseline": bool(args.also_plot_global_local_baseline),
        },
        "notes": [
            "Data source: pers_metrics.csv",
            "Per-task curves use old_task_idx > 0 and personalized_acc_old_task aggregated across sampled clients.",
            "old_task_idx=0 rows in pers_metrics are delta-only summaries and are not used for accuracy aggregation.",
            "If --include_mean is set, the mean line is recomputed from per-task aggregated local accuracies.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(generated)} figures to {out_dir}")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
