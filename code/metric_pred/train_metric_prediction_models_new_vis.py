"""Training and visualization extension for metric prediction.

This script builds on the training, evaluation, and SHAP logic in
``train_metric_prediction_models_new.py``. It organizes model-comparison,
best-model, and feature-interrelation figures and tables for analysis.

Each target is modeled independently through ``MultiOutputRegressor``. The
visualizations include cross-model comparisons, per-target results, and
feature-interrelation analyses. An ``INDEX.md`` file lists the generated
visualizations.

Usage:
    python metric_pred/train_metric_prediction_models_new_vis.py
"""

import time
from datetime import datetime
from itertools import combinations
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import train_metric_prediction_models_new as base


# =========================
# Path configuration (use an output directory dedicated to this script).
# =========================
THIS_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = THIS_DIR / "outputs"


def make_unique_output_dir(root: Path) -> Path:
    """Build a timestamped output path under root, with a suffix if needed."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = f"new_vis_{timestamp}"
    candidate = root / base_name
    if not candidate.exists():
        return candidate

    for idx in range(1, 1000):
        candidate = root / f"{base_name}_{idx:03d}"
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"无法生成唯一输出目录: {root / base_name}_xxx")


OUTPUT_DIR = make_unique_output_dir(OUTPUT_ROOT)

# Visualization root and category directories.
VIS_DIR = OUTPUT_DIR / "visualizations"
CMP_DIR = VIS_DIR / "01_model_comparison"
CMP_TABLE_DIR = CMP_DIR / "tables"
BEST_DIR = VIS_DIR / "02_best_model"
BEST_PER_TARGET_DIR = BEST_DIR / "per_target"
BEST_TABLE_DIR = BEST_DIR / "tables"
INTER_DIR = VIS_DIR / "03_feature_interrelation"
INTER_ALL_DIR = INTER_DIR / "all_features"
INTER_TOP_DIR = INTER_DIR / "top30_overall"
INTER_PER_TARGET_DIR = INTER_DIR / "per_target"

# Number of features selected for interrelation analyses.
TOP_OVERALL_FEATURES = 30
TOP_PER_TARGET_FEATURES = 30
TOP_INTERACTION_FEATURES = 30
KERNEL_INTERACTION_BACKGROUND_SIZE = 40
KERNEL_INTERACTION_EVAL_SIZE = 60
KERNEL_INTERACTION_NSAMPLES = 300
MODEL_SELECTION_METRICS_2PCT = [
    ("R2", False, 1.0),
    ("MAE", True, 1.0),
    ("RMSE", True, 1.0),
    ("Acc@5pct", False, 1.0),
    ("Acc@2pct", False, 1.0),
]
MODEL_SELECTION_DESCRIPTION_2PCT = "综合排名：R2↑、MAE↓、RMSE↓、Acc@5pct↑、Acc@2pct↑ 等权平均名次"

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def configure_base_paths():
    """Point the base module's output paths and evaluation settings to this script."""
    base.OUTPUT_DIR = OUTPUT_DIR
    base.MODEL_DIR = OUTPUT_DIR / "models"
    base.PRED_DIR = OUTPUT_DIR / "predictions"
    base.METRIC_DIR = OUTPUT_DIR / "metrics"
    base.IMPORTANCE_DIR = OUTPUT_DIR / "feature_importance"
    base.PLOT_DIR = OUTPUT_DIR / "plots"
    base.MODEL_SELECTION_METRICS = MODEL_SELECTION_METRICS_2PCT
    base.MODEL_SELECTION_DESCRIPTION = MODEL_SELECTION_DESCRIPTION_2PCT
    base.evaluate_predictions = evaluate_predictions_2pct


def evaluate_predictions_2pct(model_name, y_true_df, y_pred, split):
    records = []
    target_names = list(y_true_df.columns)
    y_true = y_true_df.values

    for idx, target in enumerate(target_names):
        yt = y_true[:, idx]
        yp = y_pred[:, idx]
        records.append({
            "model": model_name,
            "split": split,
            "target": target,
            "MAE": base.mean_absolute_error(yt, yp),
            "RMSE": float(np.sqrt(base.mean_squared_error(yt, yp))),
            "R2": base.r2_score(yt, yp),
            "Acc@5pct": base.tolerance_accuracy(yt, yp, tolerance=0.05),
            "Acc@2pct": base.tolerance_accuracy(yt, yp, tolerance=0.02),
            "Acc@1mm": base.absolute_accuracy(yt, yp, tolerance=1.0),
            "Acc@2mm": base.absolute_accuracy(yt, yp, tolerance=2.0),
        })

    records.append({
        "model": model_name,
        "split": split,
        "target": "__overall_mean__",
        "MAE": base.mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(base.mean_squared_error(y_true, y_pred))),
        "R2": base.r2_score(y_true, y_pred, multioutput="uniform_average"),
        "Acc@5pct": base.tolerance_accuracy(y_true, y_pred, tolerance=0.05),
        "Acc@2pct": base.tolerance_accuracy(y_true, y_pred, tolerance=0.02),
        "Acc@1mm": base.absolute_accuracy(y_true, y_pred, tolerance=1.0),
        "Acc@2mm": base.absolute_accuracy(y_true, y_pred, tolerance=2.0),
    })
    return records


def ensure_vis_dirs():
    for path in [
        VIS_DIR, CMP_DIR, CMP_TABLE_DIR,
        BEST_DIR, BEST_PER_TARGET_DIR, BEST_TABLE_DIR,
        INTER_DIR, INTER_ALL_DIR, INTER_TOP_DIR, INTER_PER_TARGET_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


# =========================
# Shared utilities
# =========================
def clean_feature_name(name: str) -> str:
    """Remove column prefixes and use a delta prefix to keep feature names unique."""
    text = str(name)
    if text.startswith("delta_post_pre__"):
        return "Δ" + text[len("delta_post_pre__"):]
    if text.startswith("pre__"):
        return text[len("pre__"):]
    if text.startswith("target_post__"):
        return text[len("target_post__"):]
    return text


def clean_target_name(name: str) -> str:
    return clean_feature_name(name)


def model_order_from_metrics(metrics_df: pd.DataFrame):
    """Order models by model_selection_rank, or by overall test R2 when unavailable."""
    overall = metrics_df[(metrics_df["split"] == "test") & (metrics_df["target"] == "__overall_mean__")].copy()
    if "model_selection_rank" in overall.columns and overall["model_selection_rank"].notna().any():
        overall = overall.sort_values("model_selection_rank")
    else:
        overall = overall.sort_values("R2", ascending=False)
    return overall["model"].tolist()


def _save_current_fig(path: Path, dpi: int = 300):
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight")
    pdf_path = path.with_suffix(".pdf")
    plt.savefig(pdf_path, bbox_inches="tight")
    plt.close()


def best_model_prefix(best_model_name: str) -> str:
    return f"bestmodel_{base.sanitize_filename(best_model_name)}"


def shap_plot_kind_from_method(method: str):
    """Return conservative labels for SHAP dependence plots.

    Linear explainers show additive main effects, so the colored second feature is
    context rather than evidence of a learned pairwise interaction.
    """
    text = str(method)
    if "Linear" in text or "coefficient_linear" in text:
        return {
            "slug": "top30_pairwise_shap_dependence",
            "file_prefix": "shap_dependence",
            "title": "Top30 SHAP dependence",
            "index_label": "Top30 SHAP dependence",
            "description": "linear/additive model: color shows a second feature, not a learned pairwise interaction term",
        }
    return {
        "slug": "top30_pairwise_shap_interaction_dependence",
        "file_prefix": "shap_interaction_dependence",
        "title": "Top30 SHAP interaction/dependence",
        "index_label": "Top30 SHAP interaction/dependence",
        "description": "nonlinear/model-agnostic explanation: color may reveal feature interaction or conditional dependence",
    }


# =========================
# 01. Compare models.
# =========================
def build_model_comparison(metrics_df: pd.DataFrame):
    order = model_order_from_metrics(metrics_df)
    overall = metrics_df[(metrics_df["split"] == "test") & (metrics_df["target"] == "__overall_mean__")].copy()
    overall["model"] = pd.Categorical(overall["model"], categories=order, ordered=True)
    overall = overall.sort_values("model")

    metric_cols = ["MAE", "RMSE", "R2", "Acc@5pct", "Acc@2pct", "Acc@1mm", "Acc@2mm"]

    # ---- Export comparison tables ----
    overall[["model", *metric_cols]].to_csv(
        CMP_TABLE_DIR / "test_overall_metrics_by_model.csv", index=False, encoding="utf-8-sig"
    )

    # ---- Per-metric bar charts: MAE / RMSE / R2 ----
    for metric, color, ascending in [
        ("MAE", "#E45756", True),
        ("RMSE", "#F58518", True),
        ("R2", "#4C78A8", False),
    ]:
        plot_df = overall.sort_values(metric, ascending=ascending)
        plt.figure(figsize=(11, 6))
        ax = sns.barplot(data=plot_df, x=metric, y="model", color=color)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        plt.title(f"Test Overall {metric} by Model")
        plt.xlabel(metric)
        plt.ylabel("Model")
        _save_current_fig(CMP_DIR / f"overall_{metric.lower().replace('@', '_')}_by_model.png")

    # ---- Grouped accuracy bar chart ----
    acc_long = overall.melt(id_vars="model", value_vars=["Acc@5pct", "Acc@2pct", "Acc@1mm", "Acc@2mm"],
                            var_name="metric", value_name="value")
    plt.figure(figsize=(12, 6))
    sns.barplot(data=acc_long, x="model", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.title("Test Tolerance and Millimeter Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Accuracy")
    plt.legend(title="")
    _save_current_fig(CMP_DIR / "overall_accuracy_by_model.png")

    # ---- MAE / RMSE line charts ----
    plt.figure(figsize=(13, 6))
    sns.lineplot(data=overall, x="model", y="MAE", marker="o", linewidth=2, label="MAE")
    sns.lineplot(data=overall, x="model", y="RMSE", marker="o", linewidth=2, label="RMSE")
    plt.xticks(rotation=35, ha="right")
    plt.title("Test Error Metrics (MAE / RMSE) by Model")
    plt.xlabel("Model")
    plt.ylabel("Error (original units)")
    plt.grid(axis="y", alpha=0.25)
    _save_current_fig(CMP_DIR / "overall_mae_rmse_curves.png")

    # ---- R2 / accuracy line charts ----
    plt.figure(figsize=(13, 6))
    sns.lineplot(data=overall, x="model", y="R2", marker="o", linewidth=2, label="R2")
    sns.lineplot(data=overall, x="model", y="Acc@5pct", marker="o", linewidth=2, label="Acc@5pct")
    sns.lineplot(data=overall, x="model", y="Acc@2pct", marker="o", linewidth=2, label="Acc@2pct")
    sns.lineplot(data=overall, x="model", y="Acc@1mm", marker="o", linewidth=2, label="Acc@1mm")
    sns.lineplot(data=overall, x="model", y="Acc@2mm", marker="o", linewidth=2, label="Acc@2mm")
    plt.xticks(rotation=35, ha="right")
    plt.title("Test R2 and Accuracy by Model")
    plt.xlabel("Model")
    plt.ylabel("Score")
    plt.grid(axis="y", alpha=0.25)
    _save_current_fig(CMP_DIR / "overall_r2_accuracy_curves.png")

    # ---- Multi-metric heatmap (min-max scaling is for color only; labels show original values) ----
    heat_raw = overall.set_index("model")[metric_cols]
    heat_norm = heat_raw.copy()
    for col in metric_cols:
        col_min, col_max = heat_raw[col].min(), heat_raw[col].max()
        if col_max - col_min < 1e-12:
            heat_norm[col] = 0.5
        else:
            norm = (heat_raw[col] - col_min) / (col_max - col_min)
    # Reverse the color scale for error metrics, where lower values are better.
            heat_norm[col] = (1 - norm) if col in ("MAE", "RMSE") else norm
    plt.figure(figsize=(9, max(5, 0.45 * len(heat_raw))))
    sns.heatmap(heat_norm, annot=heat_raw.values, fmt=".3f", cmap="RdYlGn",
                cbar_kws={"label": "per-metric normalized (green=better)"})
    plt.title("Test Metrics Heatmap by Model (annotated with raw values)")
    plt.ylabel("Model")
    plt.xlabel("Metric")
    _save_current_fig(CMP_DIR / "overall_metrics_heatmap.png")

    # ---- Train/test R2 generalization gap ----
    train_overall = metrics_df[(metrics_df["split"] == "train") & (metrics_df["target"] == "__overall_mean__")][["model", "R2"]]
    train_overall = train_overall.rename(columns={"R2": "train_R2"})
    gap = overall[["model", "R2"]].rename(columns={"R2": "test_R2"}).merge(train_overall, on="model", how="left")
    gap["R2_gap(train-test)"] = gap["train_R2"] - gap["test_R2"]
    gap.to_csv(CMP_TABLE_DIR / "train_test_r2_gap.csv", index=False, encoding="utf-8-sig")
    gap_long = gap.melt(id_vars="model", value_vars=["train_R2", "test_R2", "R2_gap(train-test)"],
                        var_name="metric", value_name="value")
    plt.figure(figsize=(13, 6))
    sns.barplot(data=gap_long, x="model", y="value", hue="metric")
    plt.xticks(rotation=35, ha="right")
    plt.title("Train / Test R2 and Generalization Gap")
    plt.xlabel("Model")
    plt.ylabel("R2")
    plt.legend(title="")
    _save_current_fig(CMP_DIR / "train_test_r2_gap.png")

    # ---- Per-target (model × target) heatmaps: R2 / MAE / RMSE ----
    per_target = metrics_df[(metrics_df["split"] == "test") & (metrics_df["target"] != "__overall_mean__")].copy()
    per_target["target_disp"] = per_target["target"].map(clean_target_name)
    for metric, cmap, center in [("R2", "RdYlGn", 0.0), ("MAE", "rocket_r", None), ("RMSE", "rocket_r", None)]:
        pivot = per_target.pivot(index="model", columns="target_disp", values=metric)
        pivot = pivot.reindex(order)
        plt.figure(figsize=(max(10, 1.4 * pivot.shape[1]), max(6, 0.5 * pivot.shape[0])))
        sns.heatmap(pivot, annot=True, fmt=".3f", cmap=cmap, center=center)
        plt.title(f"Test {metric} Heatmap: Model x Target")
        plt.xlabel("Target metric")
        plt.ylabel("Model")
        _save_current_fig(CMP_DIR / f"per_target_{metric.lower()}_heatmap.png")
        pivot.to_csv(CMP_TABLE_DIR / f"per_target_{metric.lower()}_model_x_target.csv", encoding="utf-8-sig")

    # ---- Normalized absolute error distribution (|error| / target standard deviation) ----
    build_normalized_error_box(order)


def build_normalized_error_box(order):
    frames = []
    for model_name in order:
        pred_path = base.PRED_DIR / f"{model_name}_test_predictions.csv"
        if not pred_path.exists():
            continue
        pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
        error_cols = [c for c in pred_df.columns if c.startswith("error__")]
        for err_col in error_cols:
            target = err_col[len("error__"):]
            true_col = f"true__{target}"
            if true_col not in pred_df.columns:
                continue
            std = pred_df[true_col].std()
            std = std if std and std > 1e-8 else 1.0
            frames.append(pd.DataFrame({
                "model": model_name,
                "target": clean_target_name(target),
                "norm_abs_error": pred_df[err_col].abs() / std,
            }))
    if not frames:
        return
    err_long = pd.concat(frames, ignore_index=True)
    err_long["model"] = pd.Categorical(err_long["model"], categories=order, ordered=True)
    err_long.to_csv(CMP_TABLE_DIR / "test_normalized_abs_error_long.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(13, 6))
    sns.boxplot(data=err_long, x="model", y="norm_abs_error", showfliers=False)
    sns.stripplot(data=err_long, x="model", y="norm_abs_error", color="black", alpha=0.15, size=2)
    plt.xticks(rotation=35, ha="right")
    plt.title("Test Normalized Absolute Error by Model (|error| / std(true))")
    plt.xlabel("Model")
    plt.ylabel("Normalized absolute error")
    _save_current_fig(CMP_DIR / "test_normalized_abs_error_box_by_model.png")


# =========================
# 02. Detailed results for the best model (per-target analyses).
# =========================
def _grid_shape(n):
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    return rows, cols


def build_best_model_normalized_error_box(best_model_name):
    pred_path = base.PRED_DIR / f"{best_model_name}_test_predictions.csv"
    if not pred_path.exists():
        print(f"[Warning] 找不到最佳模型预测文件: {pred_path}")
        return
    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")
    error_cols = [c for c in pred_df.columns if c.startswith("error__")]

    frames = []
    for err_col in error_cols:
        target = err_col[len("error__"):]
        true_col = f"true__{target}"
        if true_col not in pred_df.columns:
            continue
        std = pred_df[true_col].std()
        std = std if std and std > 1e-8 else 1.0
        frames.append(pd.DataFrame({
            "target": clean_target_name(target),
            "norm_abs_error": pred_df[err_col].abs() / std,
        }))

    if not frames:
        return
    err_long = pd.concat(frames, ignore_index=True)

    prefix = best_model_prefix(best_model_name)
    err_long.to_csv(BEST_TABLE_DIR / f"{prefix}_normalized_abs_error_by_target.csv", index=False, encoding="utf-8-sig")

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=err_long, x="target", y="norm_abs_error", showfliers=False, palette="Set3")
    sns.stripplot(data=err_long, x="target", y="norm_abs_error", color="black", alpha=0.25, size=3, jitter=0.2)
    plt.xticks(rotation=25, ha="right")
    plt.title(f"{best_model_name} Test Normalized Absolute Error by Target (|error| / std(true))")
    plt.xlabel("Target metric")
    plt.ylabel("Normalized absolute error")
    _save_current_fig(BEST_DIR / f"{prefix}_normalized_abs_error_box_by_target.png")


def build_best_model_charts(best_model_name, metrics_df, target_names):
    prefix = best_model_prefix(best_model_name)
    pred_path = base.PRED_DIR / f"{best_model_name}_test_predictions.csv"
    if not pred_path.exists():
        print(f"[Warning] 找不到最佳模型预测文件: {pred_path}")
        return
    pred_df = pd.read_csv(pred_path, encoding="utf-8-sig")

    # Per-target test metrics.
    per_target = metrics_df[(metrics_df["split"] == "test")
                            & (metrics_df["model"] == best_model_name)
                            & (metrics_df["target"] != "__overall_mean__")].copy()
    per_target["target_disp"] = per_target["target"].map(clean_target_name)
    per_target_sorted = per_target.sort_values("R2", ascending=False)
    per_target_sorted[["target_disp", "MAE", "RMSE", "R2", "Acc@5pct", "Acc@2pct", "Acc@1mm", "Acc@2mm"]].to_csv(
        BEST_TABLE_DIR / "best_model_per_target_metrics.csv", index=False, encoding="utf-8-sig"
    )

    # Plot normalized absolute errors across targets for the best model.
    build_best_model_normalized_error_box(best_model_name)

    # ---- Per-target R2 / MAE / RMSE bar charts ----
    for metric, color, ascending in [("R2", "#4C78A8", False), ("MAE", "#E45756", True), ("RMSE", "#F58518", True)]:
        plot_df = per_target.sort_values(metric, ascending=ascending)
        plt.figure(figsize=(10, max(4, 0.5 * len(plot_df))))
        ax = sns.barplot(data=plot_df, x=metric, y="target_disp", color=color)
        for container in ax.containers:
            ax.bar_label(container, fmt="%.3f", padding=3, fontsize=9)
        plt.title(f"{best_model_name} Test {metric} per Target")
        plt.xlabel(metric)
        plt.ylabel("Target metric")
        _save_current_fig(BEST_DIR / f"{prefix}_per_target_{metric.lower()}_bar.png")

    targets = list(target_names)
    rows, cols = _grid_shape(len(targets))
    r2_map = dict(zip(per_target["target"], per_target["R2"]))
    mae_map = dict(zip(per_target["target"], per_target["MAE"]))

    # ---- True-versus-predicted scatter grid ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, target in enumerate(targets):
        ax = axes[idx]
        yt = pred_df[f"true__{target}"].values
        yp = pred_df[f"pred__{target}"].values
        ax.scatter(yt, yp, alpha=0.7, edgecolor="k", linewidth=0.3)
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        ax.plot([lo, hi], [lo, hi], "r--", linewidth=1)
        ax.set_title(f"{clean_target_name(target)}\nR2={r2_map.get(target, np.nan):.3f}, MAE={mae_map.get(target, np.nan):.3f}")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
    for j in range(len(targets), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Best Model True vs Predicted: {best_model_name}", y=1.0, fontsize=14)
    _save_current_fig(BEST_DIR / f"{prefix}_true_vs_pred_grid.png")

    # ---- Residual-versus-predicted grid ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, target in enumerate(targets):
        ax = axes[idx]
        yp = pred_df[f"pred__{target}"].values
        resid = pred_df[f"error__{target}"].values  # pred - true
        ax.scatter(yp, resid, alpha=0.7, edgecolor="k", linewidth=0.3)
        ax.axhline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(clean_target_name(target))
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Residual (pred - true)")
    for j in range(len(targets), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Best Model Residual vs Predicted: {best_model_name}", y=1.0, fontsize=14)
    _save_current_fig(BEST_DIR / f"{prefix}_residual_vs_pred_grid.png")

    # ---- Residual-distribution histogram grid ----
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4.0 * rows))
    axes = np.atleast_1d(axes).ravel()
    for idx, target in enumerate(targets):
        ax = axes[idx]
        resid = pred_df[f"error__{target}"].values
        sns.histplot(resid, kde=True, ax=ax, color="#72B7B2")
        ax.axvline(0, color="red", linestyle="--", linewidth=1)
        ax.set_title(clean_target_name(target))
        ax.set_xlabel("Residual (pred - true)")
    for j in range(len(targets), len(axes)):
        axes[j].axis("off")
    fig.suptitle(f"Best Model Residual Distribution: {best_model_name}", y=1.0, fontsize=14)
    _save_current_fig(BEST_DIR / f"{prefix}_residual_hist_grid.png")

    # ---- High-resolution true-versus-predicted plots for each target ----
    for target in targets:
        yt = pred_df[f"true__{target}"].values
        yp = pred_df[f"pred__{target}"].values
        lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
        plt.figure(figsize=(6.2, 6))
        plt.scatter(yt, yp, alpha=0.75, edgecolor="k", linewidth=0.3)
        plt.plot([lo, hi], [lo, hi], "r--", linewidth=1, label="ideal y=x")
    # Fitted line.
        if len(yt) >= 2 and np.std(yt) > 1e-8:
            coef = np.polyfit(yt, yp, 1)
            xs = np.linspace(lo, hi, 100)
            plt.plot(xs, coef[0] * xs + coef[1], color="#4C78A8", linewidth=1.5, label="fit")
        plt.title(f"{best_model_name} | {clean_target_name(target)}\n"
                  f"R2={r2_map.get(target, np.nan):.3f}, MAE={mae_map.get(target, np.nan):.3f}")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.legend()
        _save_current_fig(BEST_PER_TARGET_DIR / f"{prefix}_{base.sanitize_filename(clean_target_name(target))}_true_vs_pred.png")


# =========================
# 03. Interrelations among input metrics.
# =========================
def _correlation_heatmap(corr: pd.DataFrame, title: str, out_path: Path, annot: bool):
    size = max(8, 0.45 * len(corr))
    plt.figure(figsize=(size, size * 0.92))
    sns.heatmap(corr, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                annot=annot, fmt=".2f" if annot else "",
                square=False, cbar_kws={"label": "Pearson r"},
                annot_kws={"size": 7} if annot else None)
    plt.title(title)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    _save_current_fig(out_path)


def _correlation_clustermap(corr: pd.DataFrame, title: str, out_path: Path):
    valid = corr.dropna(axis=0, how="any").dropna(axis=1, how="any")
    if valid.shape[0] < 2:
        return
    size = max(8, 0.42 * len(valid))
    grid = sns.clustermap(valid, cmap="coolwarm", center=0, vmin=-1, vmax=1,
                          figsize=(size, size), cbar_kws={"label": "Pearson r"})
    grid.fig.suptitle(title, y=1.02, fontsize=13)
    grid.savefig(out_path, dpi=300, bbox_inches="tight")
    pdf_path = out_path.with_suffix(".pdf")
    grid.savefig(pdf_path, bbox_inches="tight")
    plt.close(grid.fig)


def _linear_shap_for_target(model, target_idx, X_background, X_eval, feature_names):
    coef = np.asarray(model.coef_, dtype=float)
    intercept = np.asarray(getattr(model, "intercept_", 0.0), dtype=float).reshape(-1)
    if coef.ndim == 1:
        target_coef = coef
        target_intercept = float(intercept[0]) if len(intercept) else 0.0
    elif coef.shape[0] == len(feature_names):
        target_coef = coef[:, target_idx]
        target_intercept = float(intercept[target_idx]) if len(intercept) > target_idx else 0.0
    else:
        target_coef = coef[target_idx, :]
        target_intercept = float(intercept[target_idx]) if len(intercept) > target_idx else 0.0
    background_mean = X_background[feature_names].mean(axis=0).to_numpy(dtype=float)
    eval_values = X_eval[feature_names].to_numpy(dtype=float)
    shap_values = (eval_values - background_mean) * target_coef.reshape(1, -1)
    expected_value = target_intercept + float(np.dot(background_mean, target_coef))
    return shap_values, expected_value, "coefficient_linear_shap"


def _compute_target_shap_values(best_model, target_idx, target_name, X_train_scaled, X_test_scaled, feature_names):
    if base.shap is None:
        raise RuntimeError("shap is not installed")

    X_background = X_train_scaled.sample(min(base.SHAP_BACKGROUND_SIZE, len(X_train_scaled)), random_state=base.RANDOM_STATE)
    X_eval = X_test_scaled.sample(min(base.SHAP_EVAL_SIZE, len(X_test_scaled)), random_state=base.RANDOM_STATE).sort_index()
    single_model = base.get_single_target_model(best_model, target_idx)

    try:
        native_multioutput_coef = (not hasattr(best_model, "estimators_")) and np.asarray(getattr(single_model, "coef_", [])).ndim > 1
        if native_multioutput_coef:
            shap_values, expected_value, method = _linear_shap_for_target(single_model, target_idx, X_background, X_eval, feature_names)
        elif hasattr(single_model, "coef_"):
            explainer = base.shap.LinearExplainer(single_model, X_background)
            shap_values = explainer.shap_values(X_eval)
            expected_value = explainer.expected_value
            method = "SHAP_LinearExplainer"
        else:
            explainer = base.shap.TreeExplainer(single_model)
            shap_values = explainer.shap_values(X_eval)
            expected_value = explainer.expected_value
            method = "SHAP_TreeExplainer"
    except Exception as error:
        X_background = X_train_scaled.sample(min(KERNEL_INTERACTION_BACKGROUND_SIZE, len(X_train_scaled)), random_state=base.RANDOM_STATE)
        X_eval = X_test_scaled.sample(min(KERNEL_INTERACTION_EVAL_SIZE, len(X_test_scaled)), random_state=base.RANDOM_STATE).sort_index()
        adapter = base.SingleTargetModelAdapter(best_model, target_idx)

        def predict_fn(values):
            values_df = pd.DataFrame(values, columns=feature_names)
            return np.asarray(adapter.predict(values_df)).reshape(-1)

        print(
            f"[SHAP Dependence] {target_name}: native explainer failed ({error}); "
            f"fallback KernelExplainer, background={len(X_background)}, eval={len(X_eval)}"
        )
        explainer = base.shap.KernelExplainer(predict_fn, X_background)
        shap_values = explainer.shap_values(
            X_eval,
            nsamples=KERNEL_INTERACTION_NSAMPLES,
            l1_reg=0.0,
        )
        expected_value = explainer.expected_value
        method = "SHAP_KernelExplainer"

    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = np.asarray(shap_values)
    if shap_values.ndim == 3:
        shap_values = shap_values[:, :, 0]
    expected_value = float(np.asarray(expected_value).reshape(-1)[0])
    return shap_values, expected_value, X_eval, method


def build_per_target_feature_interactions(best_model_name, best_model, X_train_scaled, X_test_scaled, feature_names, target_names):
    if base.shap is None:
        print("[Warning] shap 未安装，跳过逐输出指标 Top30 SHAP dependence 图。")
        return

    model_imp_dir = base.IMPORTANCE_DIR / best_model_name
    disp_map = {name: clean_feature_name(name) for name in feature_names}

    for target_idx, target in enumerate(target_names):
        imp_path = model_imp_dir / f"target_{base.sanitize_filename(target)}_importance.csv"
        if not imp_path.exists():
            print(f"[Warning] 找不到目标重要性文件，跳过 SHAP dependence 图: {imp_path}")
            continue

        imp_df = pd.read_csv(imp_path, encoding="utf-8-sig")
        top_feats = [f for f in imp_df.head(TOP_INTERACTION_FEATURES)["feature"].tolist() if f in feature_names]
        if len(top_feats) < 2:
            continue

        target_disp = clean_target_name(target)
        target_dir = INTER_PER_TARGET_DIR / base.sanitize_filename(target_disp)

        try:
            shap_values, expected_value, X_eval, method = _compute_target_shap_values(
                best_model, target_idx, target, X_train_scaled, X_test_scaled, feature_names
            )
        except Exception as error:
            print(f"[Warning] {best_model_name} / {target_disp} SHAP dependence 计算失败，跳过: {error}")
            continue

        plot_kind = shap_plot_kind_from_method(method)
        dependence_dir = target_dir / plot_kind["slug"]
        dependence_dir.mkdir(parents=True, exist_ok=True)

        shap_df = pd.DataFrame(shap_values, columns=feature_names, index=X_eval.index)
        shap_df.to_csv(target_dir / "top30_dependence_eval_shap_values.csv", encoding="utf-8-sig")
        X_eval_display = X_eval.rename(columns=disp_map)
        X_eval_display.to_csv(target_dir / "top30_dependence_eval_scaled_features.csv", encoding="utf-8-sig")

        feature_to_index = {feature: idx for idx, feature in enumerate(feature_names)}
        pair_records = []
        for feature_1, feature_2 in combinations(top_feats, 2):
            index_1 = feature_to_index[feature_1]
            index_2 = feature_to_index[feature_2]
            feature_1_disp = disp_map[feature_1]
            feature_2_disp = disp_map[feature_2]
            safe_name = base.sanitize_filename(f"{feature_1_disp}_vs_{feature_2_disp}")
            out_path = dependence_dir / f"{plot_kind['file_prefix']}_{safe_name}.png"
            try:
                plt.figure(figsize=(6.6, 5.6))
                base.shap.dependence_plot(
                    ind=index_1,
                    shap_values=shap_values,
                    features=X_eval_display.values,
                    feature_names=list(X_eval_display.columns),
                    interaction_index=index_2,
                    show=False,
                )
                plt.title(
                    f"{target_disp}: {plot_kind['title']}\n"
                    f"x={feature_1_disp}, color={feature_2_disp} | {best_model_name} | {method}"
                )
                plt.tight_layout()
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                pdf_path = out_path.with_suffix(".pdf")
                plt.savefig(pdf_path, bbox_inches="tight")
                plt.close()
                pair_records.append({
                    "target": target,
                    "target_display": target_disp,
                    "plot_kind": plot_kind["index_label"],
                    "plot_kind_note": plot_kind["description"],
                    "feature_1": feature_1,
                    "feature_1_display": feature_1_disp,
                    "feature_2": feature_2,
                    "feature_2_display": feature_2_disp,
                    "shap_method": method,
                    "expected_value_scaled_target": expected_value,
                    "plot_file": str(out_path.relative_to(OUTPUT_DIR)),
                })
            except Exception as error:
                plt.close()
                print(f"[Warning] SHAP dependence plot failed for {target_disp}: {feature_1_disp} vs {feature_2_disp}: {error}")

        if pair_records:
            pd.DataFrame(pair_records).to_csv(
                target_dir / "top30_pairwise_shap_dependence_index.csv",
                index=False,
                encoding="utf-8-sig",
            )
            print(f"[{plot_kind['index_label']}] {target_disp}: saved {len(pair_records)} pairwise plots to {dependence_dir}")


def build_feature_interrelation(best_model_name, X_all, feature_names, target_names):
    # Remove prefixes from display names while keeping them unique.
    disp_map = {name: clean_feature_name(name) for name in feature_names}
    X_disp = X_all[feature_names].rename(columns=disp_map)

    # ---- (b) All input metrics ----
    corr_all = X_disp.corr(method="pearson")
    corr_all.to_csv(INTER_ALL_DIR / "all_features_correlation_matrix.csv", encoding="utf-8-sig")
    _correlation_heatmap(corr_all, f"{best_model_name} 输入指标 interrelation（全部 {corr_all.shape[0]} 个）",
                         INTER_ALL_DIR / "all_features_correlation_heatmap.png", annot=False)
    _correlation_clustermap(corr_all, f"{best_model_name} 输入指标聚类相关（全部 {corr_all.shape[0]} 个）",
                            INTER_ALL_DIR / "all_features_correlation_clustermap.png")

    # ---- (a) Top 30 metrics by overall importance ----
    agg_path = base.IMPORTANCE_DIR / best_model_name / "aggregate_importance.csv"
    if agg_path.exists():
        agg_df = pd.read_csv(agg_path, encoding="utf-8-sig")
        top_feats = agg_df.head(TOP_OVERALL_FEATURES)["feature"].tolist()
        top_feats = [f for f in top_feats if f in feature_names]
        top_disp = [disp_map[f] for f in top_feats]
        agg_df.head(TOP_OVERALL_FEATURES).to_csv(
            INTER_TOP_DIR / "top30_overall_features.csv", index=False, encoding="utf-8-sig"
        )
        corr_top = X_all[top_feats].rename(columns=disp_map).corr(method="pearson")
        corr_top = corr_top.loc[top_disp, top_disp]
        corr_top.to_csv(INTER_TOP_DIR / "top30_overall_correlation_matrix.csv", encoding="utf-8-sig")
        _correlation_heatmap(corr_top, f"{best_model_name} 重要性 Top{len(top_disp)} 输入指标 interrelation",
                             INTER_TOP_DIR / "top30_overall_correlation_heatmap.png", annot=True)
        _correlation_clustermap(corr_top, f"{best_model_name} 重要性 Top{len(top_disp)} 聚类相关",
                                INTER_TOP_DIR / "top30_overall_correlation_clustermap.png")
    else:
        print(f"[Warning] 找不到聚合重要性: {agg_path}")

    # ---- Per target: interrelations among its top important features ----
    model_imp_dir = base.IMPORTANCE_DIR / best_model_name
    for target in target_names:
        imp_path = model_imp_dir / f"target_{base.sanitize_filename(target)}_importance.csv"
        if not imp_path.exists():
            continue
        imp_df = pd.read_csv(imp_path, encoding="utf-8-sig")
        feats = [f for f in imp_df.head(TOP_PER_TARGET_FEATURES)["feature"].tolist() if f in feature_names]
        if len(feats) < 2:
            continue
        target_disp = clean_target_name(target)
        target_dir = INTER_PER_TARGET_DIR / base.sanitize_filename(target_disp)
        target_dir.mkdir(parents=True, exist_ok=True)
        imp_df.head(TOP_PER_TARGET_FEATURES).to_csv(target_dir / "top_features.csv", index=False, encoding="utf-8-sig")
        disp = [disp_map[f] for f in feats]
        corr_t = X_all[feats].rename(columns=disp_map).corr(method="pearson").loc[disp, disp]
        corr_t.to_csv(target_dir / "correlation_matrix.csv", encoding="utf-8-sig")
        _correlation_heatmap(
            corr_t,
            f"目标 {target_disp} | Top{len(disp)} 重要输入指标 interrelation（{best_model_name}）",
            target_dir / "top_features_correlation_heatmap.png",
            annot=True,
        )
        _correlation_clustermap(
            corr_t,
            f"目标 {target_disp} | Top{len(disp)} 重要输入指标聚类相关（{best_model_name}）",
            target_dir / "top_features_correlation_clustermap.png",
        )


# =========================
    # Write the category index.
# =========================
def write_index(best_model_name, dataset_info, target_names):
    prefix = best_model_prefix(best_model_name)
    lines = []
    lines.append("# 指标预测 可视化结果索引\n\n")
    lines.append("本目录在原训练脚本基础上新增了分类组织的图表与表格，便于查看模型表现、逐输出指标误差、输入指标相关结构以及逐输出指标的 Top30 两两 SHAP dependence / interaction-dependence。\n\n")

    lines.append("## 任务说明\n\n")
    lines.append("本任务是多输出指标回归：输入为 `pre__*` 与 `delta_post_pre__*` 指标，输出为多个 `target_post__*` 指标。训练时对输入和输出分别使用训练集拟合的 `StandardScaler` 标准化，模型在标准化空间学习，预测后反标准化回原始单位再计算 MAE、RMSE、R2、Acc@5pct、Acc@2pct、Acc@1mm、Acc@2mm。\n\n")
    lines.append("多输出模型中每个 `target_post__*` 都按独立输出指标解释：`MultiOutputRegressor` 使用每个输出目标一个子模型；原生多输出线性模型也按每个输出的系数/贡献单独拆解。因此本索引中的 per-target 图应理解为“某一个输出指标自己的预测规律”，不是把多个输出混成一个评分。\n\n")
    lines.append(f"最佳模型统一指最终综合排名第一的模型：**{best_model_name}**。所有新增最佳模型图文件名前缀均为 `{prefix}`，避免和模型名对不上。\n\n")
    lines.append("## 数据与模型概览\n\n")
    lines.append("| 项目 | 值 |\n|---|---|\n")
    for key, value in dataset_info.items():
        lines.append(f"| {key} | {value} |\n")
    lines.append(f"| 最佳模型 | **{best_model_name}** |\n")
    lines.append(f"| 输出目标指标数 | {len(target_names)} |\n\n")

    lines.append("## 目录分类\n\n")
    lines.append("```\n")
    lines.append(f"{OUTPUT_DIR.name}/\n")
    lines.append("├── metrics/                      原始指标表 (per-target / summary / 模型排名)\n")
    lines.append("├── predictions/                  各模型 train/test 预测结果\n")
    lines.append("├── feature_importance/<model>/   各模型逐指标 + 聚合特征重要性\n")
    lines.append("├── models/                       训练好的模型与标准化器\n")
    lines.append("├── plots/                        原脚本默认图 (R2 柱状 / R2 热力 / 最佳模型散点)\n")
    lines.append("└── visualizations/               【新增】分类可视化\n")
    lines.append("    ├── 01_model_comparison/       不同模型多指标对比\n")
    lines.append("    ├── 02_best_model/             最佳模型丰富结果图 + 逐指标分析\n")
    lines.append("    └── 03_feature_interrelation/  输入预测指标相关性 + 逐输出指标 SHAP dependence\n")
    lines.append("```\n\n")

    lines.append("## 01 不同模型对比 (`visualizations/01_model_comparison/`)\n\n")
    lines.append("这一部分只使用 `metrics/metrics_per_target.csv` 与各模型 `predictions/*_test_predictions.csv`。overall 图使用 `target == __overall_mean__` 的测试集平均指标；per-target 图使用每个 `target_post__*` 的单独指标。\n\n")
    lines.append("| 文件 | 说明 |\n|---|---|\n")
    lines.append("| overall_mae_by_model.png | 各模型测试集整体 MAE 柱状；MAE 为所有输出指标和样本的平均绝对误差，越低越好。 (有对应的pdf) |\n")
    lines.append("| overall_rmse_by_model.png | 各模型测试集整体 RMSE 柱状；对大误差更敏感，越低越好。 (有对应的pdf) |\n")
    lines.append("| overall_r2_by_model.png | 各模型测试集整体 R2 柱状；使用 `multioutput=uniform_average`，越高越好。 (有对应的pdf) |\n")
    lines.append("| overall_accuracy_by_model.png | Acc@5pct / Acc@2pct / Acc@1mm / Acc@2mm 分组柱状。 (有对应的pdf) |\n")
    lines.append("| overall_mae_rmse_curves.png | MAE / RMSE 随模型排序的折线对比，用于观察误差类指标的一致性。 (有对应的pdf) |\n")
    lines.append("| overall_r2_accuracy_curves.png | R2 / Acc 随模型排序的折线对比，用于观察解释度和各类准确率的一致性。 (有对应的pdf) |\n")
    lines.append("| overall_metrics_heatmap.png | 多指标热力图；每个指标列单独 min-max 归一化配色，包含 1mm/2mm ACC，有对应的pdf。 |\n")
    lines.append("| train_test_r2_gap.png | 训练/测试 R2 与 `train_R2 - test_R2` 泛化差，用于观察过拟合。 (有对应的pdf) |\n")
    lines.append("| per_target_r2_heatmap.png | 逐输出指标 R2 热力图，矩阵为 model × target。 (有对应的pdf) |\n")
    lines.append("| per_target_mae_heatmap.png | 逐输出指标 MAE 热力图，矩阵为 model × target。 (有对应的pdf) |\n")
    lines.append("| per_target_rmse_heatmap.png | 逐输出指标 RMSE 热力图，矩阵为 model × target。 (有对应的pdf) |\n")
    lines.append("| test_normalized_abs_error_box_by_model.png | 归一化绝对误差箱线图；每个误差为 `abs(error) / std(true target)`，用于跨不同量纲输出指标比较。 (有对应的pdf) |\n")
    lines.append("| tables/*.csv | 上述图的数据表 |\n\n")

    lines.append(f"## 02 最佳模型 {best_model_name} (`visualizations/02_best_model/`)\n\n")
    lines.append("这一部分只分析最终综合排名第一的模型。预测散点、残差图来自 `predictions/<best_model>_test_predictions.csv`；逐指标性能表来自 `metrics/metrics_per_target.csv`。\n\n")
    lines.append("| 文件 | 说明 |\n|---|---|\n")
    lines.append(f"| {prefix}_per_target_r2_bar.png | 最佳模型在每个输出指标上的测试集 R2。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_per_target_mae_bar.png | 最佳模型在每个输出指标上的测试集 MAE。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_per_target_rmse_bar.png | 最佳模型在每个输出指标上的测试集 RMSE。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_true_vs_pred_grid.png | 各输出指标的真实值 vs 预测值散点网格；红虚线为理想 `y=x`。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_residual_vs_pred_grid.png | 各输出指标的残差 `pred - true` vs 预测值，用于检查系统性偏差。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_residual_hist_grid.png | 各输出指标的残差分布直方图和 KDE 曲线。 (有对应的pdf) |\n")
    lines.append(f"| {prefix}_normalized_abs_error_box_by_target.png | 最佳模型在不同目标指标之间的归一化绝对误差箱线图，有对应的pdf。 |\n")
    lines.append(f"| per_target/{prefix}_<指标>_true_vs_pred.png | 每个输出指标单独高清散点图，含理想线和线性拟合线。 (有对应的pdf) |\n")
    lines.append("| tables/best_model_per_target_metrics.csv | 最佳模型逐指标指标表 (包含 MAE、RMSE、R2、Relative ACC, 以及 1mm/2mm ACC) |\n\n")

    lines.append("## 03 输入指标相关性与 SHAP dependence (`visualizations/03_feature_interrelation/`)\n\n")
    lines.append("这一部分分析的是输入指标之间的关系，以及输入指标对各输出指标预测的条件依赖表现。相关性图使用未标准化的训练集+测试集输入值 `X_all = concat(X_train, X_test)` 计算 Pearson 相关系数；SHAP dependence 图使用标准化测试集样本 and 最佳模型的单输出解释值计算。\n\n")
    lines.append("### top30_overall_correlation_heatmap / clustermap 是怎么来的\n\n")
    lines.append("1. 先读取 `feature_importance/<best_model>/aggregate_importance.csv`。该表把最佳模型对每个输出指标的特征重要性按输入特征聚合，核心排序列是 `mean_importance`。\n")
    lines.append("2. 取综合重要性前 30 个输入指标，即 `top30_overall_features.csv`。这里的 Top30 是“对所有输出指标平均后整体最重要的输入指标”，不是某一个输出指标单独的 Top30。\n")
    lines.append("3. 在这 30 个输入指标的原始数值上计算 Pearson 相关矩阵，保存为 `top30_overall_correlation_matrix.csv`。热力图展示每对输入指标的线性相关强弱：红色接近 +1，蓝色接近 -1，接近 0 表示线性相关弱。\n")
    lines.append("4. `top30_overall_correlation_clustermap.png` 使用同一个相关矩阵，只是额外对行列做层次聚类重排。图边上的很多连线是聚类树状图 dendrogram，表示哪些输入指标的相关模式更相似；它不是模型连接、不是因果关系，也不是 GCN 边。\n\n")
    lines.append("### 逐输出指标 Top30 SHAP dependence 怎么读\n\n")
    lines.append("每个输出指标都有自己的子模型和自己的重要性排序。脚本会读取 `target_<输出指标>_importance.csv`，取该输出指标 Top30 输入特征，并对 Top30 的所有两两组合生成 SHAP dependence 图。横轴是第一个输入指标的标准化取值，纵轴是该指标对当前输出预测的 SHAP 贡献，点的颜色是第二个输入指标的标准化取值。\n\n")
    lines.append("命名规则较保守：当最佳模型使用线性解释器时，图 and 目录命名为 `Top30 SHAP dependence`，因为线性/加性模型没有显式学习两两交互项，第二个特征只是颜色上下文；当最佳模型使用树模型或 KernelExplainer 等非线性/模型无关解释时，图 and 目录命名为 `Top30 SHAP interaction/dependence`，颜色分层或非线性形态可作为可能存在条件依赖或交互效应的线索。\n\n")
    lines.append("| 文件 | 说明 |\n|---|---|\n")
    lines.append("| all_features/all_features_correlation_heatmap.png | 全部输入指标 Pearson 相关性热力图；输入指标较多时不标注数值。 (有对应的pdf) |\n")
    lines.append("| all_features/all_features_correlation_clustermap.png | 全部输入指标相关矩阵的层次聚类图；连线为聚类树。 (有对应的pdf) |\n")
    lines.append("| top30_overall/top30_overall_features.csv | 最佳模型综合重要性 Top30 输入指标清单。 |\n")
    lines.append("| top30_overall/top30_overall_correlation_heatmap.png | 综合 Top30 输入指标 Pearson 相关性热力图，格子标注相关系数。 (有对应的pdf) |\n")
    lines.append("| top30_overall/top30_overall_correlation_clustermap.png | 同一个 Top30 相关矩阵的层次聚类视图；连线表示相关模式聚类。 (有对应的pdf) |\n")
    lines.append("| per_target/<指标>/top_features.csv | 当前输出指标自己的 Top30 输入特征清单。 |\n")
    lines.append("| per_target/<指标>/top_features_correlation_heatmap.png | 当前输出指标 Top30 输入特征之间的 Pearson 相关性。 (有对应的pdf) |\n")
    lines.append("| per_target/<指标>/top_features_correlation_clustermap.png | 当前输出指标 Top30 输入特征相关矩阵的层次聚类版本。 (新增，有对应的pdf) |\n")
    lines.append("| per_target/<指标>/top30_pairwise_shap_dependence/*.png | 线性最佳模型时输出；当前输出指标 Top30 输入特征的两两 SHAP dependence 图。 (有对应的pdf) |\n")
    lines.append("| per_target/<指标>/top30_pairwise_shap_interaction_dependence/*.png | 非线性/模型无关解释时输出；当前输出指标 Top30 输入特征的 SHAP interaction/dependence 图。 (有对应的pdf) |\n")
    lines.append("| per_target/<指标>/top30_pairwise_shap_dependence_index.csv | 当前输出指标全部两两 SHAP 图的索引表，包含图类型、两个特征名、解释方法 and 图像路径。 |\n")
    lines.append("| **/*correlation_matrix.csv | 对应 Pearson 相关系数矩阵。 |\n\n")

    (OUTPUT_DIR / "INDEX.md").write_text("".join(lines), encoding="utf-8")
    print(f"[Index] 已写出分类索引: {OUTPUT_DIR / 'INDEX.md'}")
# =========================
# Main workflow.
# =========================
def main():
    configure_base_paths()
    base.ensure_dirs()
    ensure_vis_dirs()

    train_names, test_names, X_train, X_test, y_train, y_test = base.load_dataset()
    base.save_split_datasets(train_names, test_names, X_train, X_test, y_train, y_test)
    X_train_scaled, X_test_scaled, y_train_scaled, y_test_scaled, x_scaler, y_scaler = base.scale_datasets(
        X_train, X_test, y_train, y_test
    )
    feature_names = list(X_train.columns)
    target_names = list(y_train.columns)
    X_all = pd.concat([X_train, X_test], axis=0).reset_index(drop=True)

    print("=" * 70)
    print("Metric prediction: training + rich categorized visualizations")
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    print(f"Input features: {len(feature_names)}, Output targets: {len(target_names)}")
    print(f"Output: {OUTPUT_DIR}")
    print("=" * 70)

    joblib.dump(x_scaler, base.MODEL_DIR / "input_feature_standard_scaler.joblib")
    joblib.dump(y_scaler, base.MODEL_DIR / "target_standard_scaler.joblib")

    models = base.build_models()
    all_metric_records = []
    all_importance_records = []
    all_aggregate_records = []
    model_times = {}
    test_predictions = {}
    trained_models = {}

    for model_name, search_model in models.items():
        print(f"\n[Train] {model_name}")
        start_time = time.time()
        search_model.fit(X_train_scaled, y_train_scaled)
        elapsed = time.time() - start_time
        model_times[model_name] = elapsed
        print(f"[Done] {model_name} training time: {elapsed:.2f}s")

        if hasattr(search_model, "best_params_"):
            print(f"[{model_name}] Best parameters: {search_model.best_params_}")
            model = search_model.best_estimator_
        else:
            model = search_model

        trained_models[model_name] = model

        train_pred = base.inverse_transform_targets(y_scaler, model.predict(X_train_scaled), target_names)
        test_pred = base.inverse_transform_targets(y_scaler, model.predict(X_test_scaled), target_names)
        test_predictions[model_name] = test_pred

        all_metric_records.extend(base.evaluate_predictions(model_name, y_train, train_pred, "train"))
        all_metric_records.extend(base.evaluate_predictions(model_name, y_test, test_pred, "test"))

        base.save_predictions(model_name, train_names, y_train, train_pred, "train")
        base.save_predictions(model_name, test_names, y_test, test_pred, "test")
        joblib.dump(model, base.MODEL_DIR / f"{model_name}.joblib")

        print(f"[Explain] {model_name} feature importance")
        target_importance_df, aggregate_df = base.explain_model(
            model_name, model, X_train_scaled, X_test_scaled, y_test_scaled, target_names, feature_names
        )
        all_importance_records.append(target_importance_df)
        all_aggregate_records.append(aggregate_df)

    metrics_df = pd.DataFrame(all_metric_records)
    selection_df = base.model_selection_table(metrics_df)
    metrics_df = metrics_df.merge(
        selection_df[["model", "model_selection_rank", "model_selection_score"]],
        on="model",
        how="left",
    )
    metrics_df.to_csv(base.METRIC_DIR / "metrics_per_target.csv", index=False, encoding="utf-8-sig")
    selection_df.to_csv(base.METRIC_DIR / "model_selection_ranking.csv", index=False, encoding="utf-8-sig")
    summary_df = metrics_df[metrics_df["target"] == "__overall_mean__"].copy()
    summary_df.to_csv(base.METRIC_DIR / "metrics_summary.csv", index=False, encoding="utf-8-sig")

    pd.concat(all_importance_records, ignore_index=True).to_csv(
        base.IMPORTANCE_DIR / "all_models_target_importance.csv", index=False, encoding="utf-8-sig"
    )
    pd.concat(all_aggregate_records, ignore_index=True).to_csv(
        base.IMPORTANCE_DIR / "all_models_aggregate_importance.csv", index=False, encoding="utf-8-sig"
    )

    # Keep the plots generated by the base script.
    base.plot_metrics(metrics_df)
    best_model_name = selection_df.iloc[0]["model"]
    base.plot_best_model_scatter(best_model_name, y_test, test_predictions[best_model_name])

    # Generate the categorized visualizations.
    print("\n[Vis] 01 模型对比")
    build_model_comparison(metrics_df)
    print("[Vis] 02 最佳模型丰富图表")
    build_best_model_charts(best_model_name, metrics_df, target_names)
    print("[Vis] 03 输入指标 interrelation")
    build_feature_interrelation(best_model_name, X_all, feature_names, target_names)
    print("[Vis] 04 逐输出指标 Top30 两两 SHAP dependence / interaction-dependence 图")
    build_per_target_feature_interactions(
        best_model_name,
        trained_models[best_model_name],
        X_train_scaled,
        X_test_scaled,
        feature_names,
        target_names,
    )

    best_row = summary_df[(summary_df["split"] == "test") & (summary_df["model"] == best_model_name)].iloc[0]
    dataset_info = {
        "训练样本": len(X_train),
        "测试样本": len(X_test),
        "输入特征数": len(feature_names),
        "输出指标数": len(target_names),
        "最佳模型选择策略": base.MODEL_SELECTION_DESCRIPTION,
        "最佳模型测试R2": f"{best_row['R2']:.4f}",
        "最佳模型测试MAE": f"{best_row['MAE']:.4f}",
        "最佳模型测试RMSE": f"{best_row['RMSE']:.4f}",
        "最佳模型测试Acc@1mm": f"{best_row['Acc@1mm']:.4f}",
        "最佳模型测试Acc@2mm": f"{best_row['Acc@2mm']:.4f}",
    }
    write_index(best_model_name, dataset_info, target_names)

    print("\n" + "=" * 70)
    print("[Complete] 训练 + 可视化完成")
    print(f"[Output] {OUTPUT_DIR}")
    print(f"[Best model] {best_model_name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
