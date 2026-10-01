"""Phase 5b — Hardware-efficiency benchmark (batch=1) of the Baseline and Optimised models.

For every ``variant`` × ``model`` × ``precision ∈ {fp32, fp16}`` this script
measures, on a **single GPU** with **batch = 1**:

* **Latency** (ms) — mean, std, median, P90, P95, P99, min, max — of two
  scopes:

  - ``forward``: the fused network forward pass on a fixed 1×3×640×640 input,
    timed with ``torch.cuda.Event`` pairs and a ``synchronize`` after every
    iteration (GPU time of exactly one image, no queueing). This is the
    primary "model latency" figure.
  - ``end_to_end``: the deployed prediction pipeline on a real test image
    (decoded, in host memory): ``YOLO.predict()`` with the Phase 5a settings
    (``conf`` 0.25, ``retina_masks=True``: pre-processing, inference, NMS, mask
    up-sampling to the image resolution) followed by the union of the instance
    masks copied to the host — i.e. exactly the binary mask that Phase 5a
    scores, as for the U-Net and SAM 3. Timed with ``time.perf_counter``
    around a synchronised call, since it includes CPU work that CUDA events
    cannot see. This is the "deployment latency" figure.
  - ``end_to_end_dataset``: the same pipeline once on each of the first
    ``--e2e-images`` (default 100) test images, sorted by ISIC ID (the same
    images in the three repositories), after one untimed pass over them (cuDNN
    autotuning of every input shape). Its spread reflects input-dependent cost
    (image size, number of instances), which repeating one image cannot show.

* **FPS** = 1000 / mean latency (``fps``) and 1000 / median latency
  (``fps_median``), for every scope.
* **Memory** — steady-state peak VRAM allocated and reserved by PyTorch
  during the timed iterations (``torch.cuda.max_memory_allocated``/
  ``reserved``, peak counters reset **after** warm-up), the VRAM taken by the
  weights alone, and host RAM (process RSS after loading/benchmarking and peak
  RSS from ``getrusage``). The warm-up peak is reported separately
  (``vram_peak_warmup_mb``): with ``cudnn.benchmark`` it contains cuDNN's
  algorithm-search workspaces (GBs for a 10 MB model) and is not the model's
  inference footprint. The CUDA context itself (a few hundred MB, constant per
  process) is outside PyTorch's allocator and therefore not included there;
  the **driver-level** footprint of the benchmark process is reported as
  well: ``vram_cuda_context_mb`` (device memory used after creating the
  context − before) and ``vram_process_peak_mb`` (device memory used at the
  end of the forward and end-to-end loops − before the process touched the
  GPU, the larger of the two; it contains the context, the kernels and
  everything PyTorch's caching allocator reserved), read with ``nvidia-smi``.
  This is the memory a deployment target must provide.
* **Model size** — size of ``best.pt`` on disk (Ultralytics stores final
  checkpoints in **FP16**), the theoretical FP32/FP16 weight sizes
  (fused params × 4 / × 2 bytes), parameter count (unfused and fused) and GFLOPs at
  640×640 (fused model, Ultralytics/thop convention).

Isolation & fairness:

* Every configuration runs in a **fresh subprocess** (``--worker``), so peak
  memory, allocator caches and cuDNN autotuning never leak between models.
* ``cudnn.benchmark=True`` (fixed input shape, as in deployment); recorded in
  the output. Warm-up iterations are discarded.
* GPU utilisation and memory used by *other* processes are recorded before
  each run; a busy GPU is flagged ``contended`` (latencies are then not
  publication-grade — re-run on an idle GPU).

Outputs::

    <project>/phase5_test/efficiency/<variant>_<model>_<precision>.json

(including the raw per-iteration latencies for distribution plots). Results
whose weights and settings are unchanged are skipped.

Usage:
    python benchmark_efficiency.py --project /workspace/logs/pipeline_final_v1 --device 0
    python benchmark_efficiency.py --models nano --variants optimized --precisions fp16
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from common import (
    DEFAULT_DATA_YAML,
    DEFAULT_ORDER,
    DEFAULT_PIPELINE_ROOT,
    IMGSZ,
    SEED,
    PipelinePaths,
    atomic_write_json,
    config_hash,
    read_json,
    sha256_file,
    utc_now_iso,
)

VARIANTS: tuple[str, ...] = ("baseline", "optimized")
PRECISIONS: tuple[str, ...] = ("fp32", "fp16")

#: Version of the measurement method. Part of the cache key: bump it whenever
#: what or how this script measures changes, so stale results are recomputed.
#: 3: end_to_end = final full-resolution mask on the host (retina_masks, conf);
#:    + end_to_end_dataset scope; + driver-level process VRAM.
BENCHMARK_VERSION: int = 3

#: GPU utilisation (%) above which a run is flagged as contended.
CONTENTION_UTIL_PCT: int = 5

MB: float = 1024 ** 2


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def latency_stats(ms: list[float]) -> dict[str, float]:
    """Summarise per-iteration latencies (ms) and derive FPS."""
    a = np.asarray(ms, dtype=float)
    mean, median = float(a.mean()), float(np.median(a))
    return {
        "n": int(a.size),
        "mean_ms": mean,
        "std_ms": float(a.std(ddof=1)) if a.size > 1 else 0.0,
        "median_ms": median,
        "p90_ms": float(np.percentile(a, 90)),
        "p95_ms": float(np.percentile(a, 95)),
        "p99_ms": float(np.percentile(a, 99)),
        "min_ms": float(a.min()),
        "max_ms": float(a.max()),
        "fps": 1000.0 / mean,
        "fps_median": 1000.0 / median,
    }


# ----------------------------------------------------------------------------
# Worker (runs in its own process)
# ----------------------------------------------------------------------------
def _rss_mb() -> float:
    import psutil

    return psutil.Process().memory_info().rss / MB


def _peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024  # KiB on Linux


def device_used_mb(device: str) -> float | None:
    """Memory in use on ``device`` as reported by the driver (``nvidia-smi``, MiB); ``None`` if unavailable."""
    if device == "cpu":
        return None
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits", "--query-gpu=memory.used"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout
        return float(out.strip().splitlines()[0])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def _delta(before: float | None, after: float | None) -> float | None:
    return None if before is None or after is None else after - before


def run_worker(args: argparse.Namespace) -> dict[str, Any]:
    """Benchmark one weights file at one precision; return the measurement dict."""
    import cv2
    import torch
    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops

    from segmentation_metrics import predicted_union_mask

    torch.manual_seed(SEED)
    cuda = args.device != "cpu"
    dev = torch.device(f"cuda:{args.device}" if cuda else "cpu")
    half = args.precision == "fp16"
    if half and not cuda:
        raise RuntimeError("FP16 benchmarking requires a CUDA device")
    dtype = torch.float16 if half else torch.float32
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    dev_used_start = device_used_mb(args.device)   # before this process touches the GPU
    rss_start = _rss_mb()
    if cuda:
        torch.cuda.set_device(dev)
        torch.zeros(1, device=dev)  # create the CUDA context before measuring
        torch.cuda.synchronize(dev)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)
    dev_used_ctx = device_used_mb(args.device)
    rss_ctx = _rss_mb()
    alloc0 = torch.cuda.memory_allocated(dev) if cuda else 0

    # ---- Model statistics (CPU, FP32) --------------------------------------
    yolo = YOLO(args.weights)
    net = yolo.model
    params = sum(p.numel() for p in net.parameters())
    net = net.fuse(verbose=False) if hasattr(net, "fuse") else net
    params_fused = sum(p.numel() for p in net.parameters())
    gflops = float(get_flops(net, imgsz=IMGSZ))

    # ---- Forward-pass benchmark ---------------------------------------------
    net = net.to(dev).eval()
    for p in net.parameters():
        p.requires_grad_(False)
    if half:
        net = net.half()
    weights_vram = (torch.cuda.memory_allocated(dev) - alloc0) / MB if cuda else None
    rss_model = _rss_mb()

    gen = torch.Generator(device="cpu").manual_seed(SEED)
    x = torch.rand(1, 3, IMGSZ, IMGSZ, generator=gen).to(dev, dtype)
    fwd_ms: list[float] = []
    with torch.inference_mode():
        for _ in range(args.warmup):
            net(x)
        if cuda:
            torch.cuda.synchronize(dev)
            warmup_peak = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB
            torch.cuda.reset_peak_memory_stats(dev)  # exclude cuDNN autotune workspaces
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(args.iters):
            if cuda:
                start.record()
                net(x)
                end.record()
                torch.cuda.synchronize(dev)
                fwd_ms.append(start.elapsed_time(end))
            else:
                t0 = time.perf_counter()
                net(x)
                fwd_ms.append((time.perf_counter() - t0) * 1000)
    fwd_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB if cuda else None
    fwd_peak_reserved = torch.cuda.max_memory_reserved(dev) / MB if cuda else None
    dev_used_fwd = device_used_mb(args.device)

    # ---- End-to-end prediction pipeline benchmark ---------------------------
    del net, yolo
    if cuda:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)

    def read(path: str):
        img = cv2.imread(path)
        if img is None:
            raise RuntimeError(f"cannot read test image {path}")
        return img

    image = read(args.sample_image)
    predictor = YOLO(args.weights)
    kw = dict(imgsz=IMGSZ, conf=args.conf, half=half, device=args.device, retina_masks=True, verbose=False)

    def predict(bgr):
        """Binary lesion mask at the image resolution, on the host (as scored in Phase 5a)."""
        result = predictor.predict(bgr, **kw)[0]
        return predicted_union_mask(result, *result.orig_shape)

    def timed(bgr) -> float:
        if cuda:
            torch.cuda.synchronize(dev)
        t0 = time.perf_counter()
        predict(bgr)
        if cuda:
            torch.cuda.synchronize(dev)
        return (time.perf_counter() - t0) * 1000

    for _ in range(args.e2e_warmup):
        predict(image)
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    e2e_ms = [timed(image) for _ in range(args.e2e_iters)]
    e2e_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB if cuda else None
    e2e_peak_reserved = torch.cuda.max_memory_reserved(dev) / MB if cuda else None

    # Distinct test images (decoded beforehand): one untimed pass, then one timed pass.
    ds_names = json.loads(Path(args.e2e_image_list).read_text()) if args.e2e_image_list else []
    ds_images = [read(p) for p in ds_names]
    for img in ds_images:
        predict(img)
    if cuda:
        torch.cuda.synchronize(dev)
        torch.cuda.reset_peak_memory_stats(dev)
    ds_ms = [timed(img) for img in ds_images]
    ds_peak_alloc = (torch.cuda.max_memory_allocated(dev) - alloc0) / MB if cuda and ds_images else None
    dev_used_e2e = device_used_mb(args.device)
    process_mb = [v for v in (_delta(dev_used_start, dev_used_fwd), _delta(dev_used_start, dev_used_e2e))
                  if v is not None]

    size_disk = os.path.getsize(args.weights) / MB
    return {
        "forward": {**latency_stats(fwd_ms), "timer": "torch.cuda.Event" if cuda else "perf_counter",
                    "input_shape": [1, 3, IMGSZ, IMGSZ], "raw_ms": fwd_ms},
        "end_to_end": {**latency_stats(e2e_ms), "timer": "perf_counter+synchronize",
                       "pipeline": "YOLO.predict(conf, retina_masks=True) + union mask to host",
                       "sample_image": args.sample_image, "raw_ms": e2e_ms},
        "end_to_end_dataset": {
            **latency_stats(ds_ms), "timer": "perf_counter+synchronize",
            "images": [Path(p).stem for p in ds_names],
            "image_shapes": [list(img.shape[:2]) for img in ds_images], "raw_ms": ds_ms,
        } if ds_ms else None,
        "memory": {
            "vram_weights_mb": weights_vram,
            "vram_peak_allocated_mb": fwd_peak_alloc,
            "vram_peak_reserved_mb": fwd_peak_reserved,
            "vram_peak_allocated_e2e_mb": e2e_peak_alloc,
            "vram_peak_reserved_e2e_mb": e2e_peak_reserved,
            "vram_peak_allocated_e2e_dataset_mb": ds_peak_alloc,
            "vram_peak_warmup_mb": warmup_peak if cuda else None,
            "vram_cuda_context_mb": _delta(dev_used_start, dev_used_ctx),
            "vram_process_forward_mb": _delta(dev_used_start, dev_used_fwd),
            "vram_process_e2e_mb": _delta(dev_used_start, dev_used_e2e),
            "vram_process_peak_mb": max(process_mb) if process_mb else None,
            "ram_rss_start_mb": rss_start,
            "ram_rss_after_cuda_init_mb": rss_ctx,
            "ram_rss_model_loaded_mb": rss_model,
            "ram_rss_end_mb": _rss_mb(),
            "ram_peak_rss_mb": _peak_rss_mb(),
            "ram_model_delta_mb": rss_model - rss_ctx,
            "note": "vram_peak_* from the PyTorch allocator (CUDA context excluded), steady state "
                    "after warm-up; vram_peak_warmup_mb includes cuDNN autotune workspaces; "
                    "*_e2e_* covers YOLO.predict() incl. pre/post-processing; vram_process_* and "
                    "vram_cuda_context_mb are driver-level (nvidia-smi) deltas of this process, "
                    "CUDA context and allocator cache included.",
        },
        "model": {
            "params": int(params),
            "params_fused": int(params_fused),
            "gflops": gflops,
            "size_mb_disk": size_disk,
            "size_disk_dtype": "fp16 (Ultralytics strip_optimizer)",
            "size_mb_fp32_theoretical": params_fused * 4 / MB,
            "size_mb_fp16_theoretical": params_fused * 2 / MB,
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "cudnn_benchmark": args.cudnn_benchmark,
            "gpu": torch.cuda.get_device_name(dev) if cuda else None,
            "cpu": platform.processor() or platform.machine(),
        },
    }


# ----------------------------------------------------------------------------
# Parent orchestration
# ----------------------------------------------------------------------------
def gpu_snapshot(device: str) -> dict[str, Any]:
    """Utilisation / memory of the benchmark GPU right before a run (via nvidia-smi)."""
    if device == "cpu":
        return {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "-i", str(device), "--format=csv,noheader,nounits",
             "--query-gpu=name,driver_version,utilization.gpu,memory.used,memory.total,"
             "clocks.sm,clocks.max.sm,temperature.gpu"],
            check=True, capture_output=True, text=True, timeout=30,
        ).stdout.strip().split(", ")
    except (OSError, subprocess.SubprocessError) as e:
        return {"error": str(e)}
    keys = ["name", "driver", "util_pct", "mem_used_mb", "mem_total_mb", "sm_clock_mhz",
            "sm_clock_max_mhz", "temp_c"]
    snap: dict[str, Any] = dict(zip(keys, out))
    for k in keys[2:]:
        try:
            snap[k] = float(snap[k])
        except (KeyError, ValueError):
            pass
    snap["contended"] = isinstance(snap.get("util_pct"), float) and snap["util_pct"] > CONTENTION_UTIL_PCT
    return snap


def benchmark_one(
    variant: str, model_size: str, precision: str, args: argparse.Namespace,
    paths: PipelinePaths, sample_image: str, e2e_images: list[str],
) -> dict[str, Any]:
    """Run (or skip) one configuration in a fresh worker process and save its JSON."""
    weights = paths.best_pt(variant, model_size)
    if not weights.exists():
        raise RuntimeError(f"weights not found: {weights} (run Phase 1/4 first)")
    out_json = paths.phase5_efficiency_json(variant, model_size, precision)
    settings = {
        "weights_sha256": sha256_file(weights), "precision": precision, "imgsz": IMGSZ,
        "warmup": args.warmup, "iters": args.iters, "e2e_warmup": args.e2e_warmup,
        "e2e_iters": args.e2e_iters, "cudnn_benchmark": args.cudnn_benchmark,
        "device": args.device, "sample_image": sample_image, "conf": args.conf,
        "e2e_images": [Path(i).name for i in e2e_images],
        "benchmark_version": BENCHMARK_VERSION,
    }
    previous = read_json(out_json)
    if previous and previous.get("settings_hash") == config_hash(settings) and not args.force:
        return {"tag": out_json.stem, "skipped": True, "payload": previous}

    before = gpu_snapshot(args.device)
    if before.get("contended"):
        print(f"  [warn] GPU {args.device} is busy ({before['util_pct']:.0f}% util, "
              f"{before['mem_used_mb']:.0f} MB used) — latencies will be flagged as contended")
    with tempfile.TemporaryDirectory() as tmp:
        tmp_json = Path(tmp) / "result.json"
        image_list = Path(tmp) / "e2e_images.json"
        image_list.write_text(json.dumps(e2e_images))
        cmd = [
            sys.executable, str(Path(__file__).resolve()), "--worker",
            "--weights", str(weights), "--precision", precision, "--device", args.device,
            "--sample-image", sample_image, "--out", str(tmp_json),
            "--warmup", str(args.warmup), "--iters", str(args.iters),
            "--e2e-warmup", str(args.e2e_warmup), "--e2e-iters", str(args.e2e_iters),
            "--conf", str(args.conf), "--e2e-image-list", str(image_list),
        ]
        if not args.cudnn_benchmark:
            cmd.append("--no-cudnn-benchmark")
        subprocess.run(cmd, check=True)
        result = json.loads(tmp_json.read_text())
    after = gpu_snapshot(args.device)

    payload = {
        "variant": variant, "model": model_size, "precision": precision, "batch": 1,
        "weights": str(weights), **result,
        "gpu_before": before, "gpu_after": after,
        "contended": bool(before.get("contended") or after.get("contended")),
        "settings": settings, "settings_hash": config_hash(settings), "created_at": utc_now_iso(),
    }
    atomic_write_json(out_json, payload)
    return {"tag": out_json.stem, "skipped": False, "payload": payload}


def summary_line(pl: dict[str, Any]) -> str:
    """One-line digest of a result: forward / end-to-end latency, FPS and peak VRAM."""
    fw, ee, ds, mem = pl["forward"], pl["end_to_end"], pl.get("end_to_end_dataset"), pl["memory"]
    ds_txt = f" | e2e (dataset) P95={ds['p95_ms']:.2f} ms" if ds else ""
    proc = mem.get("vram_process_peak_mb")
    return (f"fwd median={fw['median_ms']:.2f} ms P95={fw['p95_ms']:.2f} ms FPS={fw['fps']:.1f} | "
            f"e2e median={ee['median_ms']:.2f} ms P95={ee['p95_ms']:.2f} ms{ds_txt} | "
            f"VRAM peak={mem['vram_peak_allocated_mb'] or 0:.0f} MB (allocator)"
            f"{f', {proc:.0f} MB (process)' if proc is not None else ''}"
            f"{'  [CONTENDED]' if pl['contended'] else ''}")


def _test_images(data: str, n: int) -> list[str]:
    """The first ``max(n, 1)`` test images sorted by ISIC ID (stem) — the same images in every repository."""
    from train_all_models_cv import collect_test_images, load_data_yaml

    base = Path(data).resolve()
    images = sorted(collect_test_images(load_data_yaml(base), base), key=lambda p: p.stem)
    return [str(p) for p in images[:max(n, 1)]]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments (parent and ``--worker`` modes)."""
    p = argparse.ArgumentParser(description="Phase 5b — batch=1 efficiency benchmark (FP32 / FP16).")
    p.add_argument("--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER)
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=VARIANTS)
    p.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=PRECISIONS)
    p.add_argument("--data", default=DEFAULT_DATA_YAML, help="data.yaml (first test image = e2e sample).")
    p.add_argument("--device", default="0", help="Single GPU id (default: 0); 'cpu' for smoke tests.")
    p.add_argument("--project", default=DEFAULT_PIPELINE_ROOT, help="Pipeline root.")
    p.add_argument("--warmup", type=int, default=50, help="Discarded forward iterations (default: 50).")
    p.add_argument("--iters", type=int, default=500, help="Timed forward iterations (default: 500).")
    p.add_argument("--e2e-warmup", type=int, default=20, help="Discarded predict() calls (default: 20).")
    p.add_argument("--e2e-iters", type=int, default=200, help="Timed predict() calls (default: 200).")
    p.add_argument("--e2e-images", type=int, default=100,
                   help="Distinct test images timed once each in the end_to_end_dataset scope (default: 100; 0 = off).")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Confidence threshold of the instances merged into the mask (default: 0.25, as Phase 5a).")
    p.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false",
                   help="Disable cuDNN autotuning (default: enabled, as in deployment).")
    p.add_argument("--force", action="store_true", help="Re-run even if results are up to date.")
    # Worker mode (internal).
    p.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--weights", help=argparse.SUPPRESS)
    p.add_argument("--precision", choices=PRECISIONS, help=argparse.SUPPRESS)
    p.add_argument("--sample-image", help=argparse.SUPPRESS)
    p.add_argument("--e2e-image-list", help=argparse.SUPPRESS)
    p.add_argument("--out", help=argparse.SUPPRESS)
    return p.parse_args()


def main() -> int:
    """Benchmark every requested configuration (or run one worker).

    Returns:
        ``0`` on success, ``1`` if any configuration failed, ``2`` on bad usage.
    """
    args = parse_args()
    if args.worker:
        Path(args.out).write_text(json.dumps(run_worker(args)))
        return 0
    if "," in args.device:
        print("[error] Phase 5b must run on a single device (e.g. --device 0).", file=sys.stderr)
        return 2

    paths = PipelinePaths(Path(args.project))
    test_images = _test_images(args.data, args.e2e_images)
    sample_image, e2e_images = test_images[0], test_images[:args.e2e_images]
    print(f"Phase 5b — efficiency (batch=1) on device {args.device}")
    print(f"  variants={args.variants} models={args.models} precisions={args.precisions}")
    print(f"  forward: {args.warmup} warm-up + {args.iters} timed | "
          f"predict(): {args.e2e_warmup} warm-up + {args.e2e_iters} timed | "
          f"dataset: {len(e2e_images)} test images (1 untimed + 1 timed pass)")

    failures = 0
    for variant in args.variants:
        for m in args.models:
            for precision in args.precisions:
                try:
                    r = benchmark_one(variant, m, precision, args, paths, sample_image, e2e_images)
                    print(f"  [{'skip (up to date)' if r['skipped'] else 'ok'}] {r['tag']:<26} "
                          f"{summary_line(r['payload'])}")
                except Exception:
                    failures += 1
                    print(f"  [fail] {variant}/{m}/{precision}:\n{traceback.format_exc()}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
