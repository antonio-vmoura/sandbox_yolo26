"""Aggregate the WVC 2026 training curves, HPO and CV results in reproducibility/wvc2026/.

Usage: python3 analyze_wvc.py [out.json] [logs_dir]
  out.json   optional JSON dump of every statistic (the console summary is always printed)
  logs_dir   optional folder with the pipeline terminal logs (*.log, searched recursively);
             only needed for the inference-speed statistics, which are skipped without it
"""
import csv, glob, json, re, statistics as st, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # .../reproducibility/wvc2026
MODELS = ["nano", "small", "medium", "large", "xlarge"]
LETTER = {"n": "nano", "s": "small", "m": "medium", "l": "large", "x": "xlarge"}
out = {}


def read_csv(p):
    with open(p) as f:
        return [{k.strip(): v.strip() for k, v in r.items()} for r in csv.DictReader(f)]


def fl(r, k):
    return float(r[k])


def fit_m(r):  # mask-only fitness
    return 0.1 * fl(r, "metrics/mAP50(M)") + 0.9 * fl(r, "metrics/mAP50-95(M)")


def fit_bm(r):  # Ultralytics segment fitness (box + mask) used for best.pt
    return fit_m(r) + 0.1 * fl(r, "metrics/mAP50(B)") + 0.9 * fl(r, "metrics/mAP50-95(B)")


def curve_stats(path, patience):
    rows = read_csv(path)
    n = len(rows)
    b = max(rows, key=fit_bm)
    bm = max(rows, key=fit_m)
    ep = lambda r: int(float(r["epoch"]))
    tl = [fl(r, "train/seg_loss") for r in rows]
    vl = [fl(r, "val/seg_loss") for r in rows]
    vmin_i = min(range(n), key=lambda i: vl[i])
    m5095 = [fl(r, "metrics/mAP50-95(M)") for r in rows]
    # epoch-to-epoch volatility of mAP50-95(M) over the last 20 epochs
    tail = m5095[-20:]
    vol = st.pstdev([b2 - a for a, b2 in zip(tail, tail[1:])]) if len(tail) > 2 else None
    return {
        "epochs_run": n,
        "early_stopped": n < 120,
        "best_epoch_bm": ep(b), "best_epoch_m": ep(bm),
        "best": {k: fl(b, f"metrics/{k}") for k in
                 ["mAP50(M)", "mAP50-95(M)", "precision(M)", "recall(M)", "mAP50(B)", "mAP50-95(B)"]},
        "best_m_only": {k: fl(bm, f"metrics/{k}") for k in ["mAP50(M)", "mAP50-95(M)", "recall(M)"]},
        "last": {k: fl(rows[-1], f"metrics/{k}") for k in ["mAP50(M)", "mAP50-95(M)", "recall(M)"]},
        "train_seg_loss_first_last": (tl[0], tl[-1]),
        "train_seg_loss_drop_pct": 100 * (tl[0] - tl[-1]) / tl[0],
        "val_seg_loss_min": (vl[vmin_i], vmin_i + 1),
        "val_seg_loss_last": vl[-1],
        "val_seg_loss_rise_after_min_pct": 100 * (vl[-1] - vl[vmin_i]) / vl[vmin_i],
        "map5095M_best_minus_last": max(m5095) - m5095[-1],
        "map5095M_tail_step_std": vol,
        "epochs_after_best": n - ep(b),
        "patience": patience,
        "recall_eq_1_epochs": sum(1 for r in rows if abs(fl(r, "metrics/recall(M)") - 1.0) < 1e-9),
        "curve_m5095": m5095, "curve_val_seg": vl, "curve_train_seg": tl,
    }


def args(p):
    d = {}
    for line in open(p):
        if ":" in line:
            k, v = line.split(":", 1)
            d[k.strip()] = v.strip()
    return d


# ---- Phase 1 baseline / Phase 4 fine-tuning --------------------------------
for tag, pat in [("phase1", "baseline/{m}"),
                 ("finetune", "optimized/{m}")]:
    out[tag] = {}
    for m in MODELS:
        d = ROOT / pat.format(m=m)
        a = args(d / "args.yaml")
        s = curve_stats(d / "results.csv", int(a["patience"]))
        s["args"] = {k: a.get(k) for k in ["optimizer", "cos_lr", "patience", "epochs", "batch",
                                           "imgsz", "lr0", "lrf", "momentum", "weight_decay",
                                           "mosaic", "mixup", "copy_paste", "device", "amp"]}
        out[tag][m] = s

# ---- HPO -------------------------------------------------------------------
out["hpo"] = {}
for m in MODELS:
    d = ROOT / "hpo" / m
    rows = read_csv(d / "tune_results.csv")
    fit = [float(r["fitness"]) for r in rows]
    bi = max(range(len(fit)), key=lambda i: fit[i])
    best_yaml = {}
    hdr = []
    for line in open(d / "best_hyperparameters.yaml"):
        if line.startswith("#"):
            hdr.append(line.strip("# \n"))
        elif ":" in line:
            k, v = line.split(":", 1)
            best_yaml[k.strip()] = float(v)
    out["hpo"][m] = {
        "n": len(fit), "iter1_fitness": fit[0], "best_fitness": fit[bi], "best_iter": bi + 1,
        "gain_vs_iter1_pct": 100 * (fit[bi] - fit[0]) / fit[0],
        "median": st.median(fit), "min": min(fit), "max": max(fit),
        "n_within_1pct_of_best": sum(1 for f in fit if f >= 0.99 * fit[bi]),
        "best_hp": best_yaml, "iter1_hp": {k: float(v) for k, v in rows[0].items() if k != "fitness"},
        "header": hdr,
    }

# ---- Cross-validation --------------------------------------------------------
out["cv"] = {}
folds = [r["cv_fold"] for r in read_csv(ROOT / "summary/dataset_and_cv_splits.csv") if r["cv_fold"] != ""]
for m in MODELS:
    d = ROOT / "cv" / m
    rows = read_csv(d / "metrics_per_fold.csv")
    a = args(d / "fold_0/args.yaml")
    sizes = [(len(folds) - folds.count(str(k)), folds.count(str(k))) for k in range(len(rows))]
    agg = {}
    for key in ["f1_m", "f1_b", "map50_m", "map5095_m", "precision_m", "recall_m", "best_epoch"]:
        v = [float(r[key]) for r in rows]
        agg[key] = {"mean": st.mean(v), "std": st.stdev(v), "min": min(v), "max": max(v), "folds": v}
    f1 = agg["f1_m"]["mean"]
    jsi_folds = [x / (2 - x) for x in agg["f1_m"]["folds"]]
    out["cv"][m] = {
        "agg": agg, "fold_sizes": sizes,
        "jsi_from_mean_f1": f1 / (2 - f1),
        "jsi_mean_of_fold_conversions": st.mean(jsi_folds),
        "args": {k: a.get(k) for k in ["optimizer", "lr0", "mosaic", "patience", "cos_lr"]},
    }

# ---- Efficiency from terminal logs ------------------------------------------
summ_re = re.compile(r"YOLO26([nslmx])-seg summary \(fused\): (\d+) layers, ([\d,]+) parameters, \d+ gradients, ([\d.]+) GFLOPs")
speed_re = re.compile(r"^Speed: ([\d.]+)ms preprocess, ([\d.]+)ms inference, [\d.]+ms loss, ([\d.]+)ms postprocess")
eff = {m: {"inference_ms": [], "pre_ms": [], "post_ms": []} for m in MODELS}
logs = sorted(glob.glob(str(Path(sys.argv[2]) / "**/*.log"), recursive=True)) if len(sys.argv) > 2 else []
seen = set()
for p in logs:
    cur = None
    with open(p, errors="replace") as f:
        for line in f:
            line = line.replace("\r", "")
            mm = summ_re.search(line)
            if mm:
                cur = LETTER[mm.group(1)]
                eff[cur].update(layers=int(mm.group(2)), params=int(mm.group(3).replace(",", "")),
                                gflops=float(mm.group(4)))
                continue
            ms = speed_re.search(line.strip())
            if ms and cur:
                key = (p, f.tell()) if False else None
                eff[cur]["inference_ms"].append(float(ms.group(2)))
                eff[cur]["pre_ms"].append(float(ms.group(1)))
                eff[cur]["post_ms"].append(float(ms.group(3)))
                cur = None
for m in MODELS:
    v = eff[m]["inference_ms"]
    eff[m]["n_speed"] = len(v)
    if v:
        eff[m]["inference_ms_median"] = st.median(v)
        eff[m]["inference_ms_iqr"] = (sorted(v)[len(v) // 4], sorted(v)[(3 * len(v)) // 4])
        tot = [a + b + c for a, b, c in zip(eff[m]["pre_ms"], v, eff[m]["post_ms"])]
        eff[m]["fps_e2e_median"] = 1000 / st.median(tot)
    for k in ["pre_ms", "post_ms"]:
        eff[m].pop(k)
out["efficiency"] = eff

if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)

# ---- Console summary --------------------------------------------------------
P = print
P("== EFFICIENCY")
for m in MODELS:
    e = eff[m]
    P(f"{m:7} params={e.get('params',0)/1e6:6.2f}M GFLOPs={e.get('gflops')} infer_med={e.get('inference_ms_median')} "
      f"iqr={e.get('inference_ms_iqr')} n={e['n_speed']} fps={e.get('fps_e2e_median',0):.0f}")
for tag in ["phase1", "finetune"]:
    P(f"== {tag.upper()}")
    for m in MODELS:
        s = out[tag][m]
        b = s["best"]
        P(f"{m:7} ep_run={s['epochs_run']:3} best_ep(bm)={s['best_epoch_bm']:3} best_ep(m)={s['best_epoch_m']:3} "
          f"mAP50M={b['mAP50(M)']:.4f} mAP5095M={b['mAP50-95(M)']:.4f} R={b['recall(M)']:.4f} P={b['precision(M)']:.4f} | "
          f"m-only: {s['best_m_only']['mAP50(M)']:.4f}/{s['best_m_only']['mAP50-95(M)']:.4f} | "
          f"last 5095={s['last']['mAP50-95(M)']:.4f} gap={s['map5095M_best_minus_last']:.4f} | "
          f"trSegLoss -{s['train_seg_loss_drop_pct']:.0f}% valSeg min={s['val_seg_loss_min'][0]:.3f}@{s['val_seg_loss_min'][1]} "
          f"last={s['val_seg_loss_last']:.3f} (+{s['val_seg_loss_rise_after_min_pct']:.1f}%) tailstd={s['map5095M_tail_step_std']:.4f} R=1 epochs={s['recall_eq_1_epochs']}")
P("== DELTA finetune - phase1 (mAP50-95 M, mAP50 M)")
for m in MODELS:
    a, b = out["phase1"][m]["best"], out["finetune"][m]["best"]
    P(f"{m:7} d5095={b['mAP50-95(M)']-a['mAP50-95(M)']:+.4f} d50={b['mAP50(M)']-a['mAP50(M)']:+.4f}")
P("== HPO")
for m in MODELS:
    h = out["hpo"][m]
    P(f"{m:7} iter1={h['iter1_fitness']:.4f} best={h['best_fitness']:.4f}@{h['best_iter']} gain={h['gain_vs_iter1_pct']:+.2f}% "
      f"median={h['median']:.4f} min={h['min']:.4f} within1%={h['n_within_1pct_of_best']} lr0={h['best_hp'].get('lr0')} mosaic={h['best_hp'].get('mosaic')}")
P("== CV")
for m in MODELS:
    c = out["cv"][m]
    a = c["agg"]
    P(f"{m:7} F1M={a['f1_m']['mean']:.4f}±{a['f1_m']['std']:.4f} -> JSI(F/(2-F))={c['jsi_from_mean_f1']:.4f} (mean-of-folds {c['jsi_mean_of_fold_conversions']:.4f}) "
      f"mAP50M={a['map50_m']['mean']:.4f}±{a['map50_m']['std']:.4f} mAP5095M={a['map5095_m']['mean']:.4f}±{a['map5095_m']['std']:.4f} "
      f"R={a['recall_m']['mean']:.4f} best_ep={a['best_epoch']['folds']} sizes={c['fold_sizes'][0]} lr0={c['args']['lr0']}")
