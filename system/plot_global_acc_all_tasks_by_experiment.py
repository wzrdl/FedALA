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
        description="Plot global model accuracy curves on all tasks (task-wise) for each experiment."
    )
    p.add_argument("--runs_root", type=str, required=True)
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--dpi", type=int, default=180)
    p.add_argument("--title_suffix", type=str, default="")
    p.add_argument(
        "--include_mean",
        action="store_true",
        help="Overlay mean accuracy across seen tasks at each round.",
    )
    p.add_argument(
        "--rolling",
        type=int,
        default=1,
        help="Optional rolling mean window applied to plotted curves.",
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


def parse_acc_json_cell(cell) -> Dict[int, float]:
    if pd.isna(cell):
        return {}
    if isinstance(cell, dict):
        raw = cell
    else:
        try:
            raw = json.loads(cell)
        except Exception:
            return {}
    out = {}
    for k, v in raw.items():
        try:
            out[int(k)] = float(v)
        except Exception:
            continue
    return out


def title_with_suffix(base: str, suffix: str) -> str:
    suffix = (suffix or "").strip()
    return f"{base} ({suffix})" if suffix else base


def maybe_rolling(series: pd.Series, window: int) -> pd.Series:
    if window is None or window <= 1:
        return series
    return series.rolling(window=window, min_periods=1).mean()


def load_round_taskwise_long(runs_root: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    long_rows: List[Dict[str, object]] = []
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

            round_df = safe_read_csv(seed_dir / "round_metrics.csv")
            if round_df.empty:
                continue

            for col in ["global_round", "task_idx", "round_in_task", "is_task_switch_round"]:
                if col in round_df.columns:
                    round_df[col] = pd.to_numeric(round_df[col], errors="coerce")

            for _, row in round_df.iterrows():
                g_round = row.get("global_round")
                if pd.isna(g_round):
                    continue
                g_round = int(g_round)
                cur_task = row.get("task_idx")
                cur_task = int(cur_task) if pd.notna(cur_task) else None
                if pd.notna(row.get("is_task_switch_round", np.nan)) and int(row.get("is_task_switch_round", 0)) == 1:
                    switch_rows.append(
                        {
                            "exp_dir": exp_dir.name,
                            "exp_code": exp_code,
                            "seed": seed,
                            "global_round": g_round,
                            "task_idx": cur_task,
                        }
                    )

                acc_map = parse_acc_json_cell(row.get("acc_after_json"))
                if not acc_map:
                    continue
                seen_vals = []
                for t_eval, acc in sorted(acc_map.items()):
                    seen_vals.append(float(acc))
                    long_rows.append(
                        {
                            "exp_dir": exp_dir.name,
                            "exp_code": exp_code,
                            "seed": seed,
                            "global_round": g_round,
                            "current_task_idx": cur_task,
                            "eval_task_idx": int(t_eval),
                            "acc_after": float(acc),
                        }
                    )
                long_rows.append(
                    {
                        "exp_dir": exp_dir.name,
                        "exp_code": exp_code,
                        "seed": seed,
                        "global_round": g_round,
                        "current_task_idx": cur_task,
                        "eval_task_idx": 0,
                        "acc_after": float(np.mean(seen_vals)),
                        "curve_name": "mean_seen_tasks",
                    }
                )

    if not long_rows:
        raise FileNotFoundError(f"No parsable acc_after_json found under {runs_root}")
    long_df = pd.DataFrame(long_rows)
    switch_df = pd.DataFrame(switch_rows) if switch_rows else pd.DataFrame(
        columns=["exp_dir", "exp_code", "seed", "global_round", "task_idx"]
    )
    return long_df, switch_df


def aggregate_taskwise(long_df: pd.DataFrame) -> pd.DataFrame:
    d = long_df.copy()
    for c in ["global_round", "eval_task_idx", "acc_after"]:
        d[c] = pd.to_numeric(d[c], errors="coerce")
    agg = (
        d.groupby(["exp_dir", "exp_code", "global_round", "eval_task_idx"], dropna=False)["acc_after"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "acc_mean", "std": "acc_std"})
        .sort_values(["exp_dir", "eval_task_idx", "global_round"])
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
    agg_df: pd.DataFrame,
    switch_df: pd.DataFrame,
    out_dir: Path,
    dpi: int,
    title_suffix: str,
    include_mean: bool,
    rolling: int,
) -> str:
    part = agg_df[agg_df["exp_dir"] == exp_dir].copy()
    if part.empty:
        raise ValueError(f"No aggregated data for experiment: {exp_dir}")

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    task_ids = sorted(int(x) for x in pd.unique(part["eval_task_idx"]) if int(x) > 0)

    for idx, t in enumerate(task_ids):
        tdf = part[part["eval_task_idx"] == t].sort_values("global_round").copy()
        if tdf.empty:
            continue
        color = TASK_COLORS[idx % len(TASK_COLORS)]
        y = maybe_rolling(tdf["acc_mean"], rolling)
        ax.plot(
            tdf["global_round"],
            y,
            color=color,
            linewidth=2,
            label=f"Task {t}",
        )
        if (tdf["count"] > 1).any():
            err = tdf["acc_std"].fillna(0)
            ax.fill_between(tdf["global_round"], y - err, y + err, color=color, alpha=0.12)

    if include_mean:
        mdf = part[part["eval_task_idx"] == 0].sort_values("global_round").copy()
        if not mdf.empty:
            ax.plot(
                mdf["global_round"],
                maybe_rolling(mdf["acc_mean"], rolling),
                color="black",
                linewidth=2.2,
                linestyle="--",
                label="Mean(seen tasks)",
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

    ax.set_title(title_with_suffix(f"{exp_dir}: Global Accuracy on All Tasks", title_suffix))
    ax.set_xlabel("Global Communication Round")
    ax.set_ylabel("Accuracy")
    ax.grid(True, linestyle=":", alpha=0.35)
    ax.legend(frameon=False, ncol=min(3, len(task_ids) + (1 if include_mean else 0)))

    out_name = f"{exp_dir}_global_acc_all_tasks.png"
    save_fig(fig, out_dir / out_name, dpi)
    return out_name


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    if not runs_root.exists():
        raise FileNotFoundError(f"runs_root not found: {runs_root}")
    out_dir = Path(args.output_dir) if args.output_dir else (runs_root / "figures_taskwise_global")
    out_dir.mkdir(parents=True, exist_ok=True)

    long_df, switch_df = load_round_taskwise_long(runs_root)
    agg_df = aggregate_taskwise(long_df)
    sw_agg = aggregate_switches(switch_df)

    generated = []
    for exp_dir, exp_code in (
        agg_df[["exp_dir", "exp_code"]]
        .drop_duplicates()
        .sort_values(["exp_dir"])
        .itertuples(index=False, name=None)
    ):
        name = plot_one_experiment(
            exp_dir=exp_dir,
            exp_code=exp_code,
            agg_df=agg_df,
            switch_df=sw_agg,
            out_dir=out_dir,
            dpi=args.dpi,
            title_suffix=args.title_suffix,
            include_mean=args.include_mean,
            rolling=args.rolling,
        )
        generated.append(name)

    agg_df.to_csv(out_dir / "aggregated_taskwise_global_acc.csv", index=False)
    manifest = {
        "runs_root": str(runs_root),
        "output_dir": str(out_dir),
        "figures_generated": generated,
        "config": {
            "dpi": args.dpi,
            "title_suffix": args.title_suffix,
            "include_mean": bool(args.include_mean),
            "rolling": int(args.rolling),
        },
        "notes": [
            "Curves are parsed from round_metrics.csv column acc_after_json.",
            "Tasks not yet introduced are absent in acc_after_json, so their curves start when the task first appears.",
            "If --include_mean is used, the dashed line is mean accuracy over seen tasks at each round.",
        ],
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved {len(generated)} figures to {out_dir}")
    for name in generated:
        print(f" - {name}")


if __name__ == "__main__":
    main()
