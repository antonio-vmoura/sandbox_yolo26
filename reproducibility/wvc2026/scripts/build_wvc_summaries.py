#!/usr/bin/env python3
"""Rebuild the WVC 2026 summary tables in ../summary/ from the run artifacts in this folder.

Inputs (relative to reproducibility/wvc2026/):
  baseline/<scale>/, optimized/<scale>/   results.csv, args.yaml
  hpo/<scale>/                            best_hyperparameters.yaml, tune_results.csv
  cv/<scale>/                             metrics_per_fold.csv, fold_<k>/results.csv
  pixelwise/                              pixelwise_metrics.json, per_image/*.csv
  summary/model_complexity.csv            parameters / GFLOPs per scale
Outputs: summary/phase1_baseline.csv, phase2_hpo_best_params.csv, phase3_finetune.csv,
         phase4_cv_ultralytics_metrics.csv, pixelwise_main_table.csv, test_paired_comparisons.csv

Run:  python3 reproducibility/wvc2026/scripts/build_wvc_summaries.py   (CPU, ~10 s; needs numpy, pandas, scipy, pyyaml)
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]          # .../reproducibility/wvc2026
PX = ROOT / "pixelwise"
OUT = ROOT / "summary"
OUT.mkdir(exist_ok=True)

MODELS = ["nano", "small", "medium", "large", "xlarge"]


def fitness(df):
    """Ultralytics segment fitness used for best.pt: 0.1*mAP50 + 0.9*mAP50-95 for box plus mask."""
    return (0.1 * df["metrics/mAP50(B)"] + 0.9 * df["metrics/mAP50-95(B)"]
            + 0.1 * df["metrics/mAP50(M)"] + 0.9 * df["metrics/mAP50-95(M)"])


def read_results(p):
    df = pd.read_csv(p)
    df.columns = [c.strip() for c in df.columns]
    return df


def run_row(results_csv, args_yaml):
    df = read_results(results_csv)
    df["fitness"] = fitness(df)
    i = int(df["fitness"].values.argmax())            # first max, like Ultralytics (>=) keeps the last; see note below
    # Ultralytics keeps the LAST epoch among ties; ties are measure-zero with 5-decimal metrics, but be exact:
    best = df["fitness"].max()
    i = int(np.flatnonzero(np.isclose(df["fitness"].values, best))[-1])
    r = df.iloc[i]
    p, rc = r["metrics/precision(M)"], r["metrics/recall(M)"]
    args = yaml.safe_load(open(args_yaml)) if Path(args_yaml).exists() else {}
    return dict(
        epochs_run=len(df), best_epoch=int(r["epoch"]), best_fitness=round(float(best), 5),
        mAP50_B=r["metrics/mAP50(B)"], mAP5095_B=r["metrics/mAP50-95(B)"],
        mAP50_M=r["metrics/mAP50(M)"], mAP5095_M=r["metrics/mAP50-95(M)"],
        P_M=p, R_M=rc, F1_M=(2 * p * rc / (p + rc)) if p + rc else 0.0,
        P_B=r["metrics/precision(B)"], R_B=r["metrics/recall(B)"],
        train_hours=round(float(df["time"].iloc[-1]) / 3600, 3),
        optimizer=args.get("optimizer"), cos_lr=args.get("cos_lr"), patience=args.get("patience"),
        batch=args.get("batch"), imgsz=args.get("imgsz"), epochs_max=args.get("epochs"), amp=args.get("amp"),
        seed=args.get("seed"), deterministic=args.get("deterministic"), device=args.get("device"),
        lr0=args.get("lr0"), lrf=args.get("lrf"), momentum=args.get("momentum"), weight_decay=args.get("weight_decay"),
        mosaic=args.get("mosaic"), mixup=args.get("mixup"), scale=args.get("scale"), degrees=args.get("degrees"), shear=args.get("shear"),
    )


# ------------------------------------------------------------------ Phase 1 - baseline
rows = []
for m in MODELS:
    d = ROOT / "baseline" / m
    rows.append(dict(model=m, **run_row(d / "results.csv", d / "args.yaml")))
ph1 = pd.DataFrame(rows)
ph1.to_csv(OUT / "phase1_baseline.csv", index=False)

# ------------------------------------------------------------------ Phase 2 - HPO
rows = []
for m in MODELS:
    d = ROOT / "hpo" / m
    txt = (d / "best_hyperparameters.yaml").read_text()
    best = yaml.safe_load(txt)
    secs = float(re.search(r"complete ✅ \(([\d.]+)s\)", txt).group(1))
    bf = float(re.search(r"Best fitness=([\d.]+) observed at iteration (\d+)", txt).group(1))
    bi = int(re.search(r"observed at iteration (\d+)", txt).group(1))
    tr = pd.read_csv(d / "tune_results.csv")
    n_iter = int(re.search(r"# (\d+)/(\d+) iterations", txt).group(1))
    rows.append(dict(model=m, iterations_done=n_iter, tune_rows=len(tr), best_iteration=bi, best_fitness=bf,
                     hpo_hours=round(secs / 3600, 2), **{k: best[k] for k in best}))
ph2 = pd.DataFrame(rows)
ph2.to_csv(OUT / "phase2_hpo_best_params.csv", index=False)

# ------------------------------------------------------------------ Phase 3 - optimized fine-tuning
rows = []
for m in MODELS:
    d = ROOT / "optimized" / m
    rows.append(dict(model=m, **run_row(d / "results.csv", d / "args.yaml")))
ph3 = pd.DataFrame(rows)
ph3.to_csv(OUT / "phase3_finetune.csv", index=False)

# ------------------------------------------------------------------ Phase 4 - 5-fold CV (Ultralytics metrics, per fold)
rows = []
for m in MODELS:
    d = ROOT / "cv" / m
    pf = pd.read_csv(d / "metrics_per_fold.csv")
    hours = 0.0
    for f in range(5):
        hours += read_results(d / f"fold_{f}" / "results.csv")["time"].iloc[-1] / 3600
    for _, r in pf.iterrows():
        rows.append(dict(model=m, fold=int(r["fold"]), **{k: r[k] for k in pf.columns if k != "fold"}))
    row = dict(model=m, fold="mean", **{k: pf[k].mean() for k in pf.columns if k != "fold"})
    rows.append(row)
    rows.append(dict(model=m, fold="std(ddof=1)", **{k: pf[k].std(ddof=1) for k in pf.columns if k != "fold"}))
    rows.append(dict(model=m, fold="total_train_hours", best_epoch=round(hours, 2)))
ph4 = pd.DataFrame(rows)
ph4.to_csv(OUT / "phase4_cv_ultralytics_metrics.csv", index=False)

# ------------------------------------------------------------------ pixel-wise: json + independent recomputation
J = json.load(open(PX / "pixelwise_metrics.json"))
pc = lambda x: 100 * x
tab = []


def per_image(name):
    return pd.read_csv(PX / "per_image" / name)


for m in MODELS:
    base = per_image(f"baseline_val__{m}.csv"); opt = per_image(f"optimized_val__{m}.csv"); tst = per_image(f"optimized_test__{m}.csv")
    folds = [per_image(f"cv__{m}__fold_{f}.csv") for f in range(5)]
    cvj = np.array([f["jsi"].mean() for f in folds]); cvd = np.array([f["dsc"].mean() for f in folds])
    # independent recomputation from per-image CSVs vs the stored json
    for tag, df, key in [("baseline_val", base, J["baseline_val"][m]), ("optimized_val", opt, J["optimized_val"][m]),
                         ("optimized_test", tst, J["optimized_test"][m])]:
        assert abs(df["dsc"].mean() - key["dsc_mean"]) < 1e-9 and abs(df["jsi"].mean() - key["jsi_mean"]) < 1e-9, (m, tag)
    cvagg = J["cv_aggregate"][m]
    assert abs(cvd.mean() - cvagg["dsc_mean"]["mean"]) < 1e-9 and abs(cvj.std(ddof=1) - cvagg["jsi_mean"]["std_ddof1"]) < 1e-9
    # J = D/(2-D) holds per image
    assert np.allclose(tst["jsi"], tst["dsc"] / (2 - tst["dsc"]), atol=1e-6)
    tab.append(dict(
        model=m, n_val=len(base), n_test=len(tst), n_cv_total=sum(len(f) for f in folds),
        base_val_JSI=pc(base["jsi"].mean()), base_val_DSC=pc(base["dsc"].mean()),
        opt_val_JSI=pc(opt["jsi"].mean()), opt_val_DSC=pc(opt["dsc"].mean()),
        cv_JSI_mean=pc(cvj.mean()), cv_JSI_sd=pc(cvj.std(ddof=1)), cv_DSC_mean=pc(cvd.mean()), cv_DSC_sd=pc(cvd.std(ddof=1)),
        test_JSI=pc(tst["jsi"].mean()), test_DSC=pc(tst["dsc"].mean()),
        test_DSC_median=pc(tst["dsc"].median()), test_JSI_median=pc(tst["jsi"].median()),
        test_fail_dsc_lt_50=int((tst["dsc"] < 0.5).sum()), test_fail_pct=100 * float((tst["dsc"] < 0.5).mean()),
        test_zero_overlap=int((tst["inter"] == 0).sum()),
        test_lowconf_lt_0p25=int((tst["top_conf"] < 0.25).sum()),
        test_fail_and_lowconf=int(((tst["dsc"] < 0.5) & (tst["top_conf"] < 0.25)).sum()),
    ))
pxt = pd.DataFrame(tab).round(4)
cx = pd.read_csv(OUT / "model_complexity.csv")
cx["model"] = ["nano", "small", "medium", "large", "xlarge"]
pxt = pxt.merge(cx[["model", "params_unfused", "gflops_unfused", "params_fused", "gflops_fused"]], on="model")
pxt["params_M_unfused"] = (pxt["params_unfused"] / 1e6).round(2)
pxt["gap_cv_to_test_DSC"] = (pxt["cv_DSC_mean"] - pxt["test_DSC"]).round(3)
pxt.to_csv(OUT / "pixelwise_main_table.csv", index=False)

# ------------------------------------------------------------------ paired test statistics (5,000 bootstrap resamples + Wilcoxon)
rng = np.random.default_rng(0)
tst = {m: per_image(f"optimized_test__{m}.csv").set_index("id") for m in MODELS}
ids = tst["nano"].index
assert all(tst[m].index.equals(ids) for m in MODELS), "test image order differs between scales"
pair_rows = []
for i, a in enumerate(MODELS):
    for b in MODELS[i + 1:]:
        for metric in ("dsc", "jsi"):
            d = (tst[a][metric] - tst[b][metric]).values          # a - b
            idx = rng.integers(0, len(d), size=(5000, len(d)))
            bs = d[idx].mean(axis=1)
            lo, hi = np.percentile(bs, [2.5, 97.5])
            try:
                p = wilcoxon(d, alternative="two-sided").pvalue
            except ValueError:
                p = float("nan")
            pair_rows.append(dict(a=a, b=b, metric=metric.upper(), mean_diff_pts=100 * d.mean(), ci_lo_pts=100 * lo, ci_hi_pts=100 * hi,
                                  wilcoxon_p=p, ci_excludes_0=bool(lo > 0 or hi < 0), wilcoxon_p_lt_0p05=bool(p < 0.05)))
pairs = pd.DataFrame(pair_rows)
pairs.to_csv(OUT / "test_paired_comparisons.csv", index=False)
