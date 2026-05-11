import argparse
import sys
import time
import traceback
from pathlib import Path

from ultralytics import YOLO

# Mapa modelo -> arquivo de pesos. Editar aqui se mudar caminho/versão.
WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

# ----------------------------------------------------------------------------
# SEARCH SPACES — duas variantes selecionáveis via `--space`
# ----------------------------------------------------------------------------
# Cada entry: (min, max) ou (min, max, gain). Gain controla amplitude da
# mutação gaussiana — mais gain = mais agressivo. Default = 1.0.
#
# (1) `wide` — exploração inicial. Ranges baseados em:
#       * default do Ultralytics (Tuner.space)
#       * o que aprendemos nas v2-v7 (lr0 1e-3 a 2e-3 funciona; mosaic alto
#         ok; erasing alto corrompe máscara; mixup útil em moderação)
#     Use quando ainda não há informação prévia sobre o dataset.
#
# (2) `refined` — follow-up. Construído a partir das correlações Pearson
#     hp×fitness do HPO inicial do small (30 trials, ver
#     analyze_hpo_results.ipynb):
#       * DROPS (|r| < 0.05 — não há sinal estatístico):
#           scale, degrees, box, fliplr, warmup_momentum
#         (esses 5 hp ficam fixos no default Ultralytics)
#       * NARROWS (alto sinal — concentra GA na região promissora):
#           mixup       r=-0.77  [0, 0.3]   → [0.0, 0.05]
#           copy_paste  r=-0.67  [0, 0.3]   → [0.0, 0.05]
#           lr0         r=-0.45  [5e-4,5e-3]→ [1e-3, 4e-3]
#           translate   r=+0.23  [0, 0.3]   → [0.05, 0.20]
#           hsv_h       r=+0.23  [0, 0.03]  → [0.005, 0.025]
#           flipud      r=+0.20  [0, 0.5]   → [0.0, 0.10]
#           weight_decay         [0, 1e-3]  → [1e-6, 1e-4]
#           dfl                  [0.8, 3.0] → [0.8, 1.5]
#           mosaic               [0.5, 1.0] → [0.7, 1.0]
#
#     Espaço refined tem 15 hp (5 a menos), ~25% mais budget efetivo do GA
#     por iteração + ranges menores → convergência mais rápida.


SEARCH_SPACE_WIDE = {
    # ---- Aprendizado ----
    "lr0":            (5e-4, 5e-3),        # narrow ao redor do que sabemos funcionar
    "lrf":            (0.005, 0.05),       # final/initial LR ratio
    "momentum":       (0.85, 0.95, 0.3),   # default 0.937
    "weight_decay":   (0.0, 0.001),
    "warmup_epochs":  (1.0, 5.0),
    "warmup_momentum": (0.5, 0.95),

    # ---- Pesos das losses ----
    "box":            (3.0, 12.0),         # default 7.5
    "cls":            (0.2, 1.5),          # default 0.5
    "dfl":            (0.8, 3.0),          # default 1.5

    # ---- Augmentação de cor ----
    "hsv_h":          (0.0, 0.03),         # default 0.015 — narrow
    "hsv_s":          (0.3, 0.9),          # default 0.7
    "hsv_v":          (0.2, 0.7),          # default 0.4

    # ---- Augmentação geométrica ----
    "degrees":        (0.0, 30.0),         # default 0; lesão dermo é rotacionalmente simétrica
    "translate":      (0.0, 0.3),          # default 0.1
    "scale":          (0.2, 0.7),          # default 0.5
    "fliplr":         (0.0, 0.6),          # default 0.5
    "flipud":         (0.0, 0.5),          # default 0; lesão dermo permite flip vertical

    # ---- Mixing augmentations ----
    "mosaic":         (0.5, 1.0),          # default 1.0
    "mixup":          (0.0, 0.3),          # default 0
    "copy_paste":     (0.0, 0.3),          # default 0
}

SEARCH_SPACE_REFINED = {
    # ---- Aprendizado (narrowed em torno do small winner lr0=2.33e-3) ----
    "lr0":            (1e-3, 4e-3),
    "lrf":            (0.005, 0.05),       # mantém range
    "momentum":       (0.85, 0.95, 0.3),
    "weight_decay":   (1e-6, 1e-4),        # narrow (small winner=1e-5; default 5e-4 ruim)
    "warmup_epochs":  (1.0, 5.0),
    # warmup_momentum: |r|<0.05 → fixo no default 0.8 (drop do search)

    # ---- Pesos das losses (drops box; narrows dfl) ----
    # box: |r|<0.05 → fixo no default 7.5 (drop)
    "cls":            (0.2, 1.5),
    "dfl":            (0.8, 1.5),           # narrow (small winner=1.12)

    # ---- Augmentação de cor (narrow hsv_h em torno do high-signal) ----
    "hsv_h":          (0.005, 0.025),       # narrow
    "hsv_s":          (0.3, 0.9),
    "hsv_v":          (0.2, 0.7),

    # ---- Augmentação geométrica (drops degrees, scale, fliplr) ----
    # degrees: |r|<0.05 → fixo no default 0 (drop)
    "translate":      (0.05, 0.20),         # narrow (small winner=0.12)
    # scale: |r|<0.05 → fixo no default 0.5 (drop)
    # fliplr: |r|<0.05 → fixo no default 0.5 (drop)
    "flipud":         (0.0, 0.10),          # narrow (small winner≈0)

    # ---- Mixing augmentations (narrow mixup, copy_paste) ----
    "mosaic":         (0.7, 1.0),           # narrow (small winner=1.0)
    "mixup":          (0.0, 0.05),          # narrow drástico (r=-0.77)
    "copy_paste":     (0.0, 0.05),          # narrow drástico (r=-0.67)
}


def get_search_space(name: str) -> dict:
    """Retorna o search space pelo nome (`wide` | `refined`)."""
    if name == "wide":
        return SEARCH_SPACE_WIDE
    if name == "refined":
        return SEARCH_SPACE_REFINED
    raise ValueError(f"--space inválido: {name!r}. Use 'wide' ou 'refined'.")


DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="HPO sequencial nos 5 tamanhos de YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subset de modelos para tunar (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--iterations", type=int, default=30,
        help="Iterações por modelo (default: 30).",
    )
    p.add_argument(
        "--epochs", type=int, default=30,
        help="Épocas por trial (default: 30).",
    )
    p.add_argument(
        "--patience", type=int, default=10,
        help="Early-stopping patience por trial (default: 10).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml.",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' DDP). Use '0' para single-GPU.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-roda mesmo se best_hyperparameters.yaml já existir.",
    )
    p.add_argument(
        "--space", choices=["wide", "refined"], default="wide",
        help="Search space — 'wide' (20 hp, exploração inicial) ou 'refined' "
             "(15 hp, drops sem-sinal e narrows alto-sinal; recomendado para "
             "follow-up por-tamanho após uma rodada wide). Default: wide.",
    )
    return p.parse_args()


def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    return int(arg)


def tune_one_model(model_size: str, args: argparse.Namespace, device,
                   space: dict) -> dict:
    """Roda model.tune() para um único tamanho. Retorna stats da execução."""
    run_name = f"tune_isic_2018_task_1_{model_size}"
    out_dir = Path(args.project) / run_name
    best_yaml = out_dir / "best_hyperparameters.yaml"

    if best_yaml.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{best_yaml} já existe (use --force p/ re-rodar)",
            "elapsed_min": 0.0,
        }

    print("\n" + "=" * 80)
    print(f"=== TUNE START: {model_size}  ({args.iterations} iter × {args.epochs} ep)")
    print("=" * 80)

    t0 = time.perf_counter()
    model = YOLO(WEIGHTS[model_size])

    fixed_v7 = dict(
        # Infra
        data=args.data,
        project=args.project,
        name=run_name,
        task="segment",
        pretrained=True,
        imgsz=640,
        # Hardware
        device=device,
        batch=32,
        workers=8,
        cache=False,
        # Otimização (fixa do v7)
        amp=True,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=15,
        erasing=0.0,
        nbs=64,
        # Loop curto p/ HPO
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=0,
        save=True,
        plots=False,
        val=True,
        verbose=False,
    )

    model.tune(
        space=space,
        iterations=args.iterations,
        use_ray=False,
        **fixed_v7,
    )

    elapsed = (time.perf_counter() - t0) / 60
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
    }


def main() -> int:
    args = parse_args()
    device = parse_device(args.device)

    space = get_search_space(args.space)

    print(f"HPO em sequência para: {args.models}")
    print(f"  iterations/model = {args.iterations}")
    print(f"  epochs/trial     = {args.epochs}")
    print(f"  device           = {device}")
    print(f"  data             = {args.data}")
    print(f"  project          = {args.project}")
    print(f"  search space     = {args.space!r} ({len(space)} hp)")
    print(f"  force re-run     = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] {m}")
        try:
            stats = tune_one_model(m, args, device, space)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [skip] {stats['reason']}")
            else:
                print(f"  [done] {m} em {stats['elapsed_min']:.1f} min")
        except Exception as e:  # noqa: BLE001
            tb = traceback.format_exc()
            print(f"  [FAIL] {m}: {e}\n{tb}", file=sys.stderr)
            failures.append((m, str(e)))
            summary.append({
                "model": m, "skipped": False, "reason": f"FAILED: {e}",
                "elapsed_min": -1.0,
            })

    total_min = (time.perf_counter() - t_total) / 60

    # ------------------------------------------------------------------------
    # Sumário
    # ------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("=== SUMÁRIO HPO ALL MODELS")
    print("=" * 80)
    for s in summary:
        status = (
            "skipped" if s["skipped"]
            else f"FAILED ({s['reason']})" if s["elapsed_min"] < 0
            else f"{s['elapsed_min']:.1f} min"
        )
        print(f"  {s['model']:8s} : {status}")
    print(f"\nTempo total: {total_min:.1f} min  ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n{len(failures)} modelo(s) falharam:", [m for m, _ in failures])
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())