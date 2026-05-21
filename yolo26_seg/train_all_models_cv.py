"""
train_all_models_cv.py — Cross-validation (K-Fold) sequencial para todos os
tamanhos do YOLO26-seg no ISIC 2018 Task 1.

Este script é um irmão de ``train_all_models.py`` mas, em vez de um único
treino train/val, executa K treinos por modelo usando uma divisão K-Fold
determinística sobre os pares imagem+label do dataset YOLO. Cada modelo
gera um CSV com métricas por fold e um JSON com média ± desvio padrão de
mAP50, mAP50-95, precision, recall e F1-Score (Box e Mask).

Pontos-chave do design:
    * Mesma filosofia de orquestração de ``train_all_models.py``: iteração
      sequencial pelos tamanhos {nano, small, medium, large, xlarge}, com
      suporte a ``--models``, ``--force``, DDP via ``--device 0,1`` e
      mesma estrutura de logs ``--project /workspace/logs``.
    * Hiperparâmetros tunados são carregados dinamicamente de
      ``<project>/hpo/hpo_v3/tune_isic_2018_task_1_<MODEL>/best_hyperparameters.yaml``
      (mesmo padrão do script de treino base).
    * Splitting K-Fold determinístico (``seed=0``) feito sobre o pool de
      imagens train+val do ``data.yaml`` original, garantindo que nenhuma
      imagem apareça simultaneamente em train e val de um mesmo fold
      (prevenção de leakage). O conjunto ``test`` original NÃO é tocado.
    * Para cada fold é gerado um ``data.yaml`` apontando para um par de
      arquivos ``train.txt`` / ``val.txt`` (formato nativo Ultralytics),
      preservando ``nc`` e ``names`` do YAML original.

Uso:
    # Cross-validation em todos os 5 tamanhos (default):
    python train_all_models_cv.py

    # Cross-validation só no small (trial de validação do pipeline):
    python train_all_models_cv.py --models small

    # Forçar re-execução mesmo que folds já existam:
    python train_all_models_cv.py --models small --force

    docker run --gpus all -it --rm --ipc=host \\
    --user $(id -u):$(id -g) \\
    -e TORCH_HOME=/workspace/cache/torch -e HOME=/workspace/cache \\
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
    -v $(pwd)/datasets:/workspace/datasets \\
    -v $(pwd)/logs:/workspace/logs \\
    -v $(pwd)/yolo26_seg:/workspace/yolo26_seg \\
    -v $(pwd)/utils:/workspace/utils \\
    -v $(pwd)/cache:/workspace/cache \\
    -v /etc/passwd:/etc/passwd:ro -v /etc/group:/etc/group:ro \\
    yolo26_ft \\
    python /workspace/yolo26_seg/train_all_models_cv.py --models small \\
    2>&1 | tee logs/train_all_models_cv_small_v1.log
"""

import argparse
import csv
import json
import statistics
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import yaml
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIGURAÇÕES GERAIS
# ----------------------------------------------------------------------------
VERSION = "cv_v1"

DEFAULT_ORDER = ["nano", "small", "medium", "large", "xlarge"]

WEIGHTS = {
    "nano":   "/workspace/cache/yolo26n-seg.pt",
    "small":  "/workspace/cache/yolo26s-seg.pt",
    "medium": "/workspace/cache/yolo26m-seg.pt",
    "large":  "/workspace/cache/yolo26l-seg.pt",
    "xlarge": "/workspace/cache/yolo26x-seg.pt",
}

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

K_FOLDS_DEFAULT = 5
SEED_DEFAULT = 0

# Chaves de métricas relevantes em ``results.csv`` do Ultralytics (segment).
# Mantemos Box e Mask: o relatório final exibirá ambas, mas a seleção de
# melhor época usa Mask (mais relevante para segmentação).
METRIC_KEYS = {
    "precision_b":  "metrics/precision(B)",
    "recall_b":     "metrics/recall(B)",
    "map50_b":      "metrics/mAP50(B)",
    "map5095_b":    "metrics/mAP50-95(B)",
    "precision_m":  "metrics/precision(M)",
    "recall_m":     "metrics/recall(M)",
    "map50_m":      "metrics/mAP50(M)",
    "map5095_m":    "metrics/mAP50-95(M)",
}

BEST_EPOCH_KEY = "map5095_m"  # mAP50-95(M) é o critério padrão do best.pt


# ----------------------------------------------------------------------------
# UTILITÁRIOS DE K-FOLD SPLITTING
# ----------------------------------------------------------------------------
def load_data_yaml(path: Path) -> dict:
    """Carrega um ``data.yaml`` no formato Ultralytics/Roboflow."""
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(f"data.yaml vazio em {path}.")
    return data


def _resolve_split_dir(root: Path, value) -> list[Path]:
    """Resolve uma chave train/val do ``data.yaml`` em uma lista de diretórios.

    Aceita string única (relativa ao ``path:`` raiz ou absoluta) ou lista de
    strings. Retorna apenas diretórios existentes; entradas inválidas geram
    aviso mas não interrompem (mantém a tolerância do Ultralytics).
    """
    if value is None:
        return []
    candidates = value if isinstance(value, (list, tuple)) else [value]
    dirs: list[Path] = []
    for c in candidates:
        p = Path(c)
        if not p.is_absolute():
            p = (root / p).resolve()
        if p.exists():
            dirs.append(p)
        else:
            print(f"  [aviso] caminho de split não encontrado e ignorado: {p}")
    return dirs


def collect_image_label_pairs(data_yaml: dict, base_yaml_path: Path) -> list[tuple[Path, Path]]:
    """Coleta todos os pares ``(image, label)`` dos splits train+val.

    O pool é construído a partir das entradas ``train`` e ``val`` do
    ``data.yaml`` original (ignora ``test``, que fica de fora do CV).
    O label de uma imagem é resolvido trocando ``/images/`` por ``/labels/``
    e a extensão para ``.txt``, mantendo a convenção YOLO.
    """
    root = Path(data_yaml.get("path", base_yaml_path.parent)).resolve()
    image_dirs: list[Path] = []
    for split_key in ("train", "val"):
        image_dirs.extend(_resolve_split_dir(root, data_yaml.get(split_key)))

    if not image_dirs:
        raise ValueError(
            f"Não foi possível resolver nenhum diretório de imagens a partir "
            f"de {base_yaml_path}. Verifique as chaves train/val e path."
        )

    pairs: list[tuple[Path, Path]] = []
    seen: set[Path] = set()
    for img_dir in image_dirs:
        for img_path in sorted(img_dir.rglob("*")):
            if img_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if img_path in seen:
                continue
            label_path = Path(
                str(img_path).replace("/images/", "/labels/"),
            ).with_suffix(".txt")
            if not label_path.exists():
                # Sem label: imagem de background. Mantemos no pool (YOLO
                # aceita imagens sem .txt como background), mas marcamos com
                # caminho hipotético para registro.
                pass
            pairs.append((img_path, label_path))
            seen.add(img_path)

    if not pairs:
        raise ValueError(
            f"Pool de imagens vazio para CV. Verifique se {image_dirs} contém "
            f"arquivos com extensões {IMAGE_EXTENSIONS}."
        )
    return pairs


def build_kfold_splits(
    pairs: list[tuple[Path, Path]],
    k: int,
    seed: int,
) -> list[tuple[list[Path], list[Path]]]:
    """Gera ``k`` splits determinísticos sobre o pool de imagens.

    Implementação equivalente a ``sklearn.model_selection.KFold(shuffle=True,
    random_state=seed)`` porém sem dependência externa de scikit-learn:
    embaralha os índices com ``numpy.random.RandomState(seed)`` (mesmo RNG
    usado pela sklearn) e distribui em ``k`` blocos consecutivos onde os
    primeiros ``n % k`` blocos têm um elemento extra. Mesmo seed → mesmos
    conjuntos por fold (verificado contra ``KFold`` ref.).
    """
    if k < 2:
        raise ValueError(f"k_folds deve ser >= 2 (recebido: {k}).")
    n = len(pairs)
    if n < k:
        raise ValueError(
            f"Pool com {n} imagens é menor que k={k}. "
            f"Reduza --k-folds ou aumente o dataset.",
        )
    images = [p[0] for p in pairs]

    rng = np.random.RandomState(seed)
    indices = np.arange(n)
    rng.shuffle(indices)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1

    splits: list[tuple[list[Path], list[Path]]] = []
    start = 0
    for size in fold_sizes:
        stop = start + size
        val_idx = indices[start:stop]
        train_idx = np.concatenate([indices[:start], indices[stop:]])
        splits.append(
            ([images[i] for i in train_idx], [images[i] for i in val_idx]),
        )
        start = stop
    return splits


def write_fold_dataset(
    fold_dir: Path,
    train_images: list[Path],
    val_images: list[Path],
    template_yaml: dict,
) -> Path:
    """Escreve ``train.txt``, ``val.txt`` e ``data.yaml`` para um fold.

    O ``data.yaml`` gerado preserva ``nc``/``names`` do template original e
    aponta ``train``/``val`` para os arquivos de listagem absolutos, formato
    nativamente aceito pelo Ultralytics.
    """
    fold_dir.mkdir(parents=True, exist_ok=True)
    train_txt = fold_dir / "train.txt"
    val_txt = fold_dir / "val.txt"
    train_txt.write_text("\n".join(str(p) for p in train_images) + "\n")
    val_txt.write_text("\n".join(str(p) for p in val_images) + "\n")

    fold_yaml = fold_dir / "data.yaml"
    payload: dict = {
        "path": str(fold_dir.resolve()),
        "train": str(train_txt.resolve()),
        "val": str(val_txt.resolve()),
    }
    if "nc" in template_yaml:
        payload["nc"] = template_yaml["nc"]
    if "names" in template_yaml:
        payload["names"] = template_yaml["names"]
    with fold_yaml.open("w") as f:
        yaml.safe_dump(payload, f, sort_keys=False)
    return fold_yaml


# ----------------------------------------------------------------------------
# UTILITÁRIOS DE MÉTRICAS
# ----------------------------------------------------------------------------
def load_tuned_hp(path: Path) -> dict:
    """Carrega ``best_hyperparameters.yaml`` gerado pelo Ultralytics Tuner."""
    with path.open("r") as f:
        data = yaml.safe_load(f) or {}
    if not data:
        raise ValueError(
            f"YAML vazio em {path}. Verifique se a afinação anterior falhou.",
        )
    return data


def parse_best_metrics(results_csv: Path) -> dict[str, float]:
    """Lê o ``results.csv`` e devolve as métricas da melhor época.

    A melhor época é selecionada pelo critério padrão do Ultralytics
    (``metrics/mAP50-95(M)`` para segmentação). F1-Score é calculado a
    partir de precision e recall (Box e Mask) com a fórmula 2PR/(P+R).
    """
    rows: list[dict] = []
    with results_csv.open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.strip(): v for k, v in row.items()})
    if not rows:
        raise ValueError(f"results.csv vazio: {results_csv}")

    best_key_csv = METRIC_KEYS[BEST_EPOCH_KEY]
    best_row = max(rows, key=lambda r: float(r.get(best_key_csv, "0") or 0.0))

    out: dict[str, float] = {}
    for short, full in METRIC_KEYS.items():
        out[short] = float(best_row.get(full, "0") or 0.0)
    # F1 derivado (Box e Mask)
    for suffix in ("b", "m"):
        p = out[f"precision_{suffix}"]
        r = out[f"recall_{suffix}"]
        out[f"f1_{suffix}"] = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    out["best_epoch"] = float(best_row.get("epoch", "0") or 0.0)
    return out


def aggregate_fold_metrics(per_fold: list[dict[str, float]]) -> dict[str, dict[str, float]]:
    """Recebe lista de dicts (uma entrada por fold) e devolve mean/std por chave."""
    summary: dict[str, dict[str, float]] = {}
    if not per_fold:
        return summary
    keys = sorted(per_fold[0].keys())
    for k in keys:
        values = [m.get(k, 0.0) for m in per_fold]
        summary[k] = {
            "mean": statistics.mean(values),
            "std": statistics.pstdev(values) if len(values) > 1 else 0.0,
        }
    return summary


def save_metrics_artifacts(
    model_size: str,
    per_fold: list[dict[str, float]],
    summary: dict[str, dict[str, float]],
    out_dir: Path,
) -> tuple[Path, Path]:
    """Persiste métricas por fold (CSV) e agregadas (JSON)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics_per_fold.csv"
    json_path = out_dir / "metrics_summary.json"

    if per_fold:
        fieldnames = ["fold", *sorted(per_fold[0].keys())]
        with csv_path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for i, m in enumerate(per_fold):
                row = {"fold": i, **m}
                w.writerow(row)

    payload = {
        "model": model_size,
        "version": VERSION,
        "n_folds": len(per_fold),
        "per_fold": per_fold,
        "summary": summary,
    }
    with json_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return csv_path, json_path


# ----------------------------------------------------------------------------
# LÓGICA DE TREINO POR FOLD
# ----------------------------------------------------------------------------
def parse_device(arg: str):
    if "," in arg:
        return [int(x) for x in arg.split(",")]
    if arg == "cpu":
        return "cpu"
    return int(arg)


def train_one_fold(
    weights: str,
    fold_yaml: Path,
    base_kwargs: dict,
    run_name: str,
    project: Path,
) -> Path:
    """Treina um único fold e devolve o ``save_dir`` do run."""
    model = YOLO(weights)
    kwargs = {
        **base_kwargs,
        "data": str(fold_yaml),
        "project": str(project),
        "name": run_name,
        "exist_ok": False,
    }
    model.train(**kwargs)
    save_dir = Path(model.trainer.save_dir)
    return save_dir


def cross_validate_one_model(
    model_size: str,
    args: argparse.Namespace,
    device,
    data_yaml: dict,
    pairs: list[tuple[Path, Path]],
) -> dict:
    """Roda K-Fold CV completo em um único tamanho de modelo."""
    cv_root = Path(args.project) / "cv" / VERSION / f"yolo26_{model_size}_cv_isic_2018"
    splits_dir = cv_root / "splits"
    runs_dir = cv_root / "runs"
    metrics_dir = cv_root  # csv/json gravados na raiz do CV do modelo

    summary_json = metrics_dir / "metrics_summary.json"
    if summary_json.exists() and not args.force:
        return {
            "model": model_size,
            "skipped": True,
            "reason": f"{summary_json} já existe (use --force p/ re-executar)",
            "elapsed_min": 0.0,
        }

    hp_yaml = (
        Path(args.project)
        / "hpo"
        / "hpo_v3"
        / f"tune_isic_2018_task_1_{model_size}"
        / "best_hyperparameters.yaml"
    )
    if not hp_yaml.exists():
        return {
            "model": model_size,
            "skipped": True,
            "reason": (
                f"YAML de hiperparâmetros não encontrado: {hp_yaml}. "
                f"Execute primeiro a afinação para este modelo."
            ),
            "elapsed_min": 0.0,
        }

    weights_path = args.weights_override or WEIGHTS[model_size]

    tuned_hp = load_tuned_hp(hp_yaml)
    splits = build_kfold_splits(pairs, k=args.k_folds, seed=args.seed)

    print("\n" + "=" * 80)
    print(f"=== CROSS-VALIDATION {VERSION}: {model_size}")
    print(f"  k_folds      = {args.k_folds}   seed = {args.seed}")
    print(f"  pool_size    = {len(pairs)} imagens (train+val do data.yaml original)")
    print(f"  HP Origem    = {hp_yaml}")
    print(f"  weights      = {weights_path}")
    print(f"  cv_root      = {cv_root}")
    print("  Hiperparâmetros carregados:")
    for k, v in sorted(tuned_hp.items()):
        print(f"    {k:18s} = {v}")
    print("=" * 80)

    base_kwargs = dict(
        task="segment",
        pretrained=True,
        imgsz=args.imgsz,
        device=device,
        batch=args.batch,
        workers=args.workers,
        cache=False,
        amp=False,
        optimizer="MuSGD",
        cos_lr=True,
        close_mosaic=10,
        erasing=0.4,
        nbs=64,
        epochs=args.epochs,
        patience=args.patience,
        deterministic=True,
        seed=args.seed,
        save=True,
        plots=True,
        val=True,
        verbose=True,
    )
    base_kwargs = {**base_kwargs, **tuned_hp}

    t0 = time.perf_counter()
    per_fold_metrics: list[dict[str, float]] = []

    for k, (train_imgs, val_imgs) in enumerate(splits):
        fold_dir = splits_dir / f"fold_{k}"
        fold_yaml = write_fold_dataset(fold_dir, train_imgs, val_imgs, data_yaml)

        run_name = f"fold_{k}"
        existing_csv = runs_dir / run_name / "results.csv"
        if existing_csv.exists() and not args.force:
            print(f"\n[fold {k}/{args.k_folds - 1}] resultado já existe — pulando treino")
            metrics = parse_best_metrics(existing_csv)
        else:
            print(
                f"\n[fold {k}/{args.k_folds - 1}] "
                f"train={len(train_imgs)} val={len(val_imgs)} → {fold_yaml}",
            )
            save_dir = train_one_fold(
                weights=weights_path,
                fold_yaml=fold_yaml,
                base_kwargs=base_kwargs,
                run_name=run_name,
                project=runs_dir,
            )
            metrics = parse_best_metrics(save_dir / "results.csv")
        per_fold_metrics.append(metrics)
        # Log curto após cada fold
        print(
            f"  fold {k}: "
            f"mAP50(M)={metrics['map50_m']:.4f} "
            f"mAP50-95(M)={metrics['map5095_m']:.4f} "
            f"P(M)={metrics['precision_m']:.4f} "
            f"R(M)={metrics['recall_m']:.4f} "
            f"F1(M)={metrics['f1_m']:.4f}",
        )

    summary = aggregate_fold_metrics(per_fold_metrics)
    csv_path, json_path = save_metrics_artifacts(
        model_size, per_fold_metrics, summary, metrics_dir,
    )

    elapsed = (time.perf_counter() - t0) / 60
    print(f"\n  [{model_size}] artefatos: {csv_path}  |  {json_path}")
    return {
        "model": model_size,
        "skipped": False,
        "reason": None,
        "elapsed_min": elapsed,
        "summary": summary,
        "csv_path": str(csv_path),
        "json_path": str(json_path),
    }


# ----------------------------------------------------------------------------
# CLI & MAIN
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cross-validation K-Fold sequencial nos modelos YOLO26-seg.",
    )
    p.add_argument(
        "--models", nargs="+", default=DEFAULT_ORDER, choices=DEFAULT_ORDER,
        help=f"Subconjunto de modelos para rodar CV (default: {DEFAULT_ORDER}).",
    )
    p.add_argument(
        "--data",
        default="/workspace/datasets/isic_2018_task1_yolo26/data.yaml",
        help="Path do data.yaml original (deve apontar para train/val/test YOLO).",
    )
    p.add_argument(
        "--device", default="0,1",
        help="GPUs (default: '0,1' DDP). Use '0' p/ single-GPU ou 'cpu'.",
    )
    p.add_argument(
        "--project", default="/workspace/logs",
        help="Diretório raiz dos logs (default: /workspace/logs).",
    )
    p.add_argument(
        "--k-folds", type=int, default=K_FOLDS_DEFAULT,
        help=f"Número de folds (default: {K_FOLDS_DEFAULT}).",
    )
    p.add_argument(
        "--seed", type=int, default=SEED_DEFAULT,
        help=f"Semente determinística para o split (default: {SEED_DEFAULT}).",
    )
    p.add_argument(
        "--epochs", type=int, default=120,
        help="Épocas por fold (default: 120, mesmo do train_all_models.py).",
    )
    p.add_argument(
        "--patience", type=int, default=25,
        help="Early-stopping patience (default: 25).",
    )
    p.add_argument(
        "--batch", type=int, default=16,
        help="Batch size por fold (default: 16).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Tamanho de imagem (default: 640).",
    )
    p.add_argument(
        "--workers", type=int, default=4,
        help="Dataloader workers (default: 4).",
    )
    p.add_argument(
        "--weights-override", default=None,
        help="Substitui o caminho dos pesos do modelo (uso de smoke-test).",
    )
    p.add_argument(
        "--force", action="store_true",
        help="Re-executa folds/modelos mesmo se artefatos já existirem.",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    device = parse_device(args.device)

    base_yaml = Path(args.data).resolve()
    if not base_yaml.exists():
        print(f"[erro] data.yaml não encontrado: {base_yaml}")
        return 2
    data_yaml = load_data_yaml(base_yaml)
    pairs = collect_image_label_pairs(data_yaml, base_yaml)

    print(f"Orquestração de Cross-Validation {VERSION} para: {args.models}")
    print(f"  device       = {device}")
    print(f"  data         = {base_yaml}")
    print(f"  project      = {args.project}")
    print(f"  k_folds      = {args.k_folds}   seed = {args.seed}")
    print(f"  pool         = {len(pairs)} imagens")
    print(f"  force re-run = {args.force}")

    summary: list[dict] = []
    failures: list[tuple[str, str]] = []
    t_total = time.perf_counter()

    for i, m in enumerate(args.models, 1):
        print(f"\n[{i}/{len(args.models)}] A processar CV do modelo: {m}")
        try:
            stats = cross_validate_one_model(m, args, device, data_yaml, pairs)
            summary.append(stats)
            if stats["skipped"]:
                print(f"  [ignorado] {stats['reason']}")
            else:
                print(f"  [concluído] tempo total do CV: {stats['elapsed_min']:.1f} min")
        except Exception:
            tb = traceback.format_exc()
            print(f"  [falha] exceção no modelo {m}:\n{tb}")
            summary.append({
                "model": m, "skipped": False, "reason": "fail",
                "elapsed_min": 0.0,
            })
            failures.append((m, tb))

    print("\n" + "=" * 80)
    print(f"=== SUMÁRIO DA CROSS-VALIDATION {VERSION}")
    print("=" * 80)
    for s in summary:
        if s.get("skipped"):
            print(f"  {s['model']:<8} : ignorado ({s['reason']})")
        elif s.get("reason") == "fail":
            print(f"  {s['model']:<8} : FALHOU")
        else:
            agg = s.get("summary", {})

            def fmt(key: str, _agg: dict = agg) -> str:
                m = _agg.get(key, {})
                return f"{m.get('mean', 0):.4f}±{m.get('std', 0):.4f}"

            print(
                f"  {s['model']:<8} : sucesso  ({s['elapsed_min']:.1f} min) | "
                f"mAP50(M)={fmt('map50_m')}  "
                f"mAP50-95(M)={fmt('map5095_m')}  "
                f"P(M)={fmt('precision_m')}  "
                f"R(M)={fmt('recall_m')}  "
                f"F1(M)={fmt('f1_m')}",
            )

    total_min = (time.perf_counter() - t_total) / 60
    print(f"\nTempo total de processamento: {total_min:.1f} min ({total_min / 60:.2f} h)")

    if failures:
        print(f"\n[!] Encontrada(s) {len(failures)} falha(s):")
        for m, _ in failures:
            print(f"    - Modelo {m}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
