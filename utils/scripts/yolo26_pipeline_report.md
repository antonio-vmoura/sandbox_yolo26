# YOLO26-seg End-to-End Pipeline — ISIC 2018 Task 1
## Relatório técnico para redação de artigo científico

**Repositório:** [`antonio-vmoura/sandbox_yolo26`](https://github.com/antonio-vmoura/sandbox_yolo26)
**PRs entregues:** [#23](https://github.com/antonio-vmoura/sandbox_yolo26/pull/23) (mergeado) · [#24](https://github.com/antonio-vmoura/sandbox_yolo26/pull/24) (mergeado — isolamento dos logs)
**Versão do pipeline:** `pipeline_e2e_v1`

---

## 1. Visão geral

O objetivo do pipeline é fornecer uma execução **reprodutível, idempotente e auditável** do estudo completo de fine-tuning do **Ultralytics YOLO26-seg** para **segmentação de instâncias de lesões cutâneas** no benchmark **ISIC 2018 Task 1**, varrendo os cinco tamanhos arquiteturais da família — `nano`, `small`, `medium`, `large`, `xlarge`. O fluxo é dividido em quatro fases sequenciais com responsabilidades bem definidas:

| Fase | Objetivo científico | Script principal |
|---|---|---|
| 1 — *Baseline* | Estabelecer linha-base com **estritamente os hiperparâmetros padrão do Ultralytics**, sem otimização nem augmentations customizadas. Serve como referência mínima defensável para o artigo. | `yolo26_seg/train_baseline_models.py` (novo) |
| 2 — *HPO* | Para cada arquitetura individualmente, encontrar via **algoritmo genético embutido no `model.tune()`** o conjunto ótimo de hiperparâmetros usando um *search space* **refinado** já calibrado em sessão anterior. | `yolo26_seg/tune_all_models_v2.py` (reaproveitado da sessão `21d1...`) |
| 3 — *Optimized single-split* | Fine-tuning completo de cada arquitetura com os hiperparâmetros tunados da Fase 2, no mesmo split train/val/test original. | `yolo26_seg/train_all_models.py` (reaproveitado) |
| 4 — *5-Fold Cross-Validation* | Avaliar a **robustez estatística** dos modelos otimizados via K-Fold determinístico, produzindo média ± desvio-padrão de mAP@50, mAP@50-95, Precision, Recall e F1-Score. | `yolo26_seg/train_all_models_cv.py` (reaproveitado da sessão `d1aa...`) + `yolo26_seg/consolidate_cv_results.py` (novo) |

O orquestrador master é o shell script **`run_pipeline.sh`** na raiz do repositório, que executa as 4 fases sequencialmente, gera logs por fase + um log mestre, e produz artefatos consolidados (`pipeline_summary/`) prontos para serem citados em tabelas do artigo.

### Princípios de design
1. **Reprodutibilidade absoluta:** todos os scripts passam `deterministic=True` e `seed=0` ao Ultralytics, e o split K-Fold é determinístico via `numpy.random.RandomState(0)` (equivalente bit-a-bit a `sklearn.model_selection.KFold(shuffle=True, random_state=0)` — sem dependência de scikit-learn).
2. **Idempotência:** cada fase detecta artefatos canônicos pré-existentes (`weights/best.pt`, `best_hyperparameters.yaml`, `metrics_summary.json`) e pula trabalho redundante. Re-execuções incrementais são seguras; `--force` re-executa tudo.
3. **Isolamento de logs:** todo o output desta execução é gravado em `logs/pipeline_e2e_v1/` (configurável via `PIPELINE_NAME`), separando-o por completo dos fine-tunings anteriores que ainda residem direto em `logs/`. Ablações futuras só precisam trocar o `PIPELINE_NAME`.
4. **Containerização total:** todo o pipeline roda dentro do container Docker `yolo26_ft` definido no `Dockerfile` do repo (CUDA 12.1 + Python 3.11 + PyTorch + Ultralytics), garantindo paridade entre estação local e servidor remoto.

---

## 2. Fase 1 — Baseline (`train_baseline_models.py`)

### 2.1 Motivação científica
Antes de qualquer otimização, é fundamental documentar a performance "fora-da-caixa" para que ganhos posteriores sejam atribuíveis exclusivamente à HPO. Para isso, a Fase 1 evita **deliberadamente** sobrescrever otimizador, *learning rate scheduler*, *AMP*, augmentations agressivas ou pesos das losses — usa apenas o que o YOLO26 entrega por padrão.

### 2.2 Configuração
Argumentos passados ao `model.train()`:

| Argumento | Valor | Justificativa |
|---|---|---|
| `epochs` | `120` | Mesmo orçamento das Fases 3 e 4 → comparabilidade direta. |
| `patience` | `20` | *Early stopping* conforme especificação do estudo. |
| `imgsz` | `640` | Resolução padrão YOLO. |
| `deterministic` | `True` | Reprodutibilidade. |
| `seed` | `0` | Reprodutibilidade. |
| `pretrained` | `True` | Carrega pesos COCO pré-treinados da família YOLO26. |
| `task` | `"segment"` | Instance segmentation. |
| Demais hiperparâmetros (optimizer, lr, augmentations, etc.) | **defaults do Ultralytics** | Não sobrescritos intencionalmente. |

### 2.3 Pesos pré-treinados
A constante `WEIGHTS` no script aponta para os checkpoints COCO baixados em `/workspace/cache/yolo26{n,s,m,l,x}-seg.pt`. Se não existirem localmente, o próprio Ultralytics faz o download automático na primeira execução.

### 2.4 Outputs
Para cada modelo `<m>`:
```
logs/pipeline_e2e_v1/phase1_baseline/yolo26_<m>_baseline/
├── weights/{best.pt, last.pt}
├── results.csv                 ← curva de treino época-a-época
├── args.yaml                   ← hiperparâmetros efetivamente usados
├── confusion_matrix*.png
├── train_batch*.jpg, val_batch*.jpg
└── PR_curve.png, F1_curve.png, ...
```

### 2.5 Coleta de métricas
Após a Fase 1, o orquestrador invoca `collect_phase_metrics.py --phase baseline`, que percorre cada `results.csv`, seleciona a melhor época pelo critério padrão do Ultralytics (**`metrics/mAP50-95(M)`**, máscara), calcula F1-Score derivado (`2·P·R/(P+R)` para Box e Mask), e grava:

```
logs/pipeline_e2e_v1/pipeline_summary/baseline_metrics.csv
logs/pipeline_e2e_v1/pipeline_summary/baseline_metrics.json
```

com uma linha por modelo. Colunas: `model`, `results_csv`, `best_epoch`, `precision_b`, `recall_b`, `map50_b`, `map5095_b`, `f1_b`, `precision_m`, `recall_m`, `map50_m`, `map5095_m`, `f1_m`.

---

## 3. Fase 2 — Hyperparameter Optimization (`tune_all_models_v2.py`)

### 3.1 Motivação científica
O `model.tune()` do Ultralytics implementa um **algoritmo genético** que, a cada iteração, treina o modelo por um número curto de épocas com um conjunto candidato de HPs, avalia em validação, e refina a próxima geração via mutação/crossover ao redor das melhores soluções. O Ultralytics rastreia o histórico completo em `tune_results.csv` e grava o melhor conjunto encontrado em `best_hyperparameters.yaml`.

### 3.2 Espaço de busca — `refined`
Definido em `tune_all_models_v2.py` na constante `SEARCH_SPACE_REFINED`, herdada da sessão `21d1...`. Limites já estreitados após uma rodada exploratória anterior (`wide`):

| Categoria | Hiperparâmetros (intervalos) |
|---|---|
| Aprendizado | `lr0 ∈ [1e-3, 4e-3]`, `lrf ∈ [0.005, 0.05]`, `momentum ∈ [0.85, 0.95]` (μ=0.3), `weight_decay ∈ [1e-6, 1e-4]`, `warmup_epochs ∈ [1, 5]` |
| Pesos das losses | `cls ∈ [0.2, 1.5]`, `dfl ∈ [0.8, 1.5]` |
| Augmentação de cor | `hsv_h ∈ [0.005, 0.025]`, `hsv_s ∈ [0.3, 0.9]`, `hsv_v ∈ [0.2, 0.7]` |
| Augmentação geométrica | `translate ∈ [0.05, 0.20]`, `flipud ∈ [0.0, 0.10]` |
| Mixing | `mosaic ∈ [0.7, 1.0]`, `mixup ∈ [0.0, 0.05]`, `copy_paste ∈ [0.0, 0.05]` |

### 3.3 Configuração fixa por trial (fixed_v7)
Argumentos *não* tunados, mantidos constantes entre trials:
- `optimizer="MuSGD"`, `cos_lr=True`, `amp=True`, `close_mosaic=15`, `erasing=0.0`, `nbs=64`
- `epochs=30` por trial (loop curto), `patience=10` (default do pipeline; configurável via `HPO_PATIENCE`)
- `imgsz=640`, `batch=32`, `workers=8`, `deterministic=True`, `seed=0`

### 3.4 Volume de busca
Padrões do pipeline: **30 iterações × 30 épocas/trial × 5 modelos = 4 500 epochs** de busca total no espaço `refined`. Configurável via `HPO_ITERATIONS` e `HPO_EPOCHS_PER_TRIAL`.

### 3.5 Outputs
O orquestrador força `--project /workspace/logs/pipeline_e2e_v1/hpo/hpo_v3` para que o output caia no caminho esperado pelas Fases 3 e 4:

```
logs/pipeline_e2e_v1/hpo/hpo_v3/tune_isic_2018_task_1_<m>/
├── best_hyperparameters.yaml    ← consumido pelas Fases 3 e 4
├── tune_results.csv             ← histórico completo do GA
├── tune_fitness.png             ← curva de fitness ao longo das iterações
├── tune_scatter_plots.png       ← visualização do espaço de busca
└── train1/, train2/, ...        ← um diretório por trial
```

---

## 4. Fase 3 — Optimized Single-Split (`train_all_models.py`)

### 4.1 Motivação científica
Com o `best_hyperparameters.yaml` em mãos, treinamos cada arquitetura por orçamento completo de épocas no mesmo split train/val/test original — produzindo os modelos "finais" que serão comparados ao baseline.

### 4.2 Configuração base + tuned HPs
O script monta o dicionário `base` (linhas 149-174 de `train_all_models.py`):

| Argumento | Valor |
|---|---|
| `epochs` | `120` |
| `patience` | `25` |
| `optimizer` | `"MuSGD"` |
| `cos_lr` | `True` |
| `amp` | `False` (treino completo, sem mixed-precision para máxima precisão numérica) |
| `close_mosaic` | `10` |
| `erasing` | `0.4` |
| `nbs` | `64` |
| `batch` | `16` |
| `imgsz` | `640` |
| `deterministic`, `seed` | `True`, `0` |

E faz **`train_kwargs = {**base, **tuned_hp}`**, ou seja: os hiperparâmetros tunados sobrescrevem os valores base quando conflitam. Isso permite que a Fase 2 ajuste lr0, momentum, weight_decay, pesos das losses e augmentations, mas mantém otimizador, schedule e batch consistentes.

### 4.3 Outputs
```
logs/pipeline_e2e_v1/yolo26_<m>_ft_isic_2018_v11/
├── weights/{best.pt, last.pt}
├── results.csv
├── args.yaml                    ← inclui os HPs tunados aplicados
└── (mesmas figuras da Fase 1)
```

`v11` é a constante `VERSION` interna ao script, mantida para continuidade com PRs anteriores; agora vive dentro do parente isolado `pipeline_e2e_v1/`, sem colidir com runs antigos.

### 4.4 Coleta de métricas
O orquestrador chama `collect_phase_metrics.py --phase optimized`, produzindo `pipeline_summary/optimized_metrics.{csv,json}` com a mesma estrutura do baseline. **Esses dois arquivos são suficientes para a primeira tabela do artigo** (baseline vs. otimizado, por arquitetura).

---

## 5. Fase 4 — 5-Fold Cross-Validation (`train_all_models_cv.py` + `consolidate_cv_results.py`)

### 5.1 Motivação científica
Métricas em single-split sofrem variância amostral significativa em datasets pequenos como o ISIC 2018 Task 1. K-Fold Cross-Validation fornece estimativas mais robustas (média ± desvio-padrão) e permite reportar **intervalo de confiança** das métricas no artigo.

### 5.2 Construção do pool
O script lê o `data.yaml` original e une **train + val** num único pool de pares `(imagem, label)` (linhas 136-179). O split `test` original **não é tocado** — fica reservado para avaliação final fora do CV.

### 5.3 Splitting determinístico (sem scikit-learn)
A função `build_kfold_splits()` (linhas 182-223) implementa K-Fold manualmente:
1. `numpy.random.RandomState(seed=0)` embaralha os índices.
2. Os blocos têm tamanho `n // k`, e os primeiros `n % k` blocos recebem um elemento extra (mesmo comportamento de `sklearn.KFold`).
3. Para cada fold `k`, gera-se `train.txt` e `val.txt` com caminhos absolutos das imagens, mais um `data.yaml` por fold apontando para esses listings (formato nativo Ultralytics, preservando `nc` e `names` do template).

**Garantia:** mesmo `seed` ⇒ mesmos folds, reproduzível bit-a-bit entre execuções e independente de versão do scikit-learn.

### 5.4 Treino por fold
Para cada modelo × fold:
- Carrega `best_hyperparameters.yaml` da Fase 2 (caminho `hpo/hpo_v3/.../tune_isic_2018_task_1_<m>/`).
- Mescla com o mesmo `base_kwargs` da Fase 3 (epochs=120, patience=25, MuSGD, cos_lr, AMP off).
- Treina e identifica a melhor época pelo critério `metrics/mAP50-95(M)`.
- Extrai 8 métricas brutas (Precision, Recall, mAP50, mAP50-95 — Box e Mask) e calcula F1-Score derivado.

### 5.5 Agregação per-modelo
Após os K folds, `aggregate_fold_metrics()` calcula média (`statistics.mean`) e desvio-padrão **populacional** (`statistics.pstdev`) de cada métrica, e grava:

```
logs/pipeline_e2e_v1/cv/cv_v1/yolo26_<m>_cv_isic_2018/
├── splits/fold_{0..4}/{train.txt, val.txt, data.yaml}
├── runs/fold_{0..4}/{weights, results.csv, args.yaml, ...}
├── metrics_per_fold.csv         ← uma linha por fold
└── metrics_summary.json         ← per_fold + summary (mean/std)
```

### 5.6 Consolidação global (novo: `consolidate_cv_results.py`)
Roda automaticamente após a Fase 4. Percorre os 5 `metrics_summary.json` e gera:

```
logs/pipeline_e2e_v1/pipeline_summary/cv_consolidated.csv
logs/pipeline_e2e_v1/pipeline_summary/cv_consolidated.json
```

- **CSV:** uma linha por modelo, colunas `<metric>_mean` + `<metric>_std` para cada uma das 10 métricas reportadas (5× Box + 5× Mask) + `best_epoch_mean/std` + `n_folds`. Pronto para colar em LaTeX como tabela do artigo.
- **JSON:** mesmo conteúdo plus `per_fold` (útil para gerar boxplots / violin plots no notebook de análise).

---

## 6. Estrutura final dos artefatos

Após uma execução completa do pipeline em todos os 5 modelos, este é o layout produzido:

```
logs/
├── pipeline_e2e_v1/                                  ← isolado por PIPELINE_NAME
│   ├── phase1_baseline/
│   │   ├── yolo26_nano_baseline/{weights, results.csv, ...}
│   │   ├── yolo26_small_baseline/...
│   │   ├── yolo26_medium_baseline/...
│   │   ├── yolo26_large_baseline/...
│   │   └── yolo26_xlarge_baseline/...
│   ├── hpo/hpo_v3/
│   │   ├── tune_isic_2018_task_1_nano/{best_hyperparameters.yaml, tune_results.csv, ...}
│   │   ├── tune_isic_2018_task_1_small/...
│   │   ├── tune_isic_2018_task_1_medium/...
│   │   ├── tune_isic_2018_task_1_large/...
│   │   └── tune_isic_2018_task_1_xlarge/...
│   ├── yolo26_nano_ft_isic_2018_v11/                 ← Fase 3 (otimizado, single-split)
│   ├── yolo26_small_ft_isic_2018_v11/
│   ├── yolo26_medium_ft_isic_2018_v11/
│   ├── yolo26_large_ft_isic_2018_v11/
│   ├── yolo26_xlarge_ft_isic_2018_v11/
│   ├── cv/cv_v1/                                     ← Fase 4
│   │   ├── yolo26_nano_cv_isic_2018/{splits, runs/fold_{0..4}, metrics_per_fold.csv, metrics_summary.json}
│   │   ├── yolo26_small_cv_isic_2018/...
│   │   ├── yolo26_medium_cv_isic_2018/...
│   │   ├── yolo26_large_cv_isic_2018/...
│   │   └── yolo26_xlarge_cv_isic_2018/...
│   ├── pipeline_summary/                             ← artefatos consolidados para o artigo
│   │   ├── baseline_metrics.{csv,json}
│   │   ├── optimized_metrics.{csv,json}
│   │   └── cv_consolidated.{csv,json}
│   └── pipeline_runs/<UTC-timestamp>/                ← logs do orquestrador
│       ├── pipeline.log
│       ├── phase1.log, phase1_collect.log
│       ├── phase2.log
│       ├── phase3.log, phase3_collect.log
│       └── phase4.log, phase4_consolidate.log
│
├── pipeline_e2e_v2/                                  ← ablações futuras (basta trocar PIPELINE_NAME)
└── <runs antigos preservados intactos>               ← fine-tunings anteriores
```

---

## 7. Como executar o pipeline

### 7.1 Pré-requisitos
- Host Linux com Docker e **NVIDIA Container Toolkit** instalados.
- GPUs NVIDIA (2× recomendado para DDP; 1× funciona).
- Dataset ISIC 2018 Task 1 já convertido para o formato YOLO+Roboflow em `./datasets/isic_2018_task1_yolo26/data.yaml` (estrutura `train/images`, `train/labels`, `valid/images`, `valid/labels`, etc.).
- `./cache/` com os pesos pré-treinados (`yolo26{n,s,m,l,x}-seg.pt`) — o Ultralytics baixa automaticamente se faltar.

### 7.2 Build do container (1ª vez)

```bash
cd /caminho/para/sandbox_yolo26
docker build -t yolo26_ft .
```

### 7.3 Verificação rápida sem GPU (dry-run)

Antes de gastar GPU, simule o grafo de comandos para confirmar paths/flags:

```bash
LOGS_ROOT=/tmp/yolo26_dryrun \
  bash run_pipeline.sh --dry-run
```

A saída lista cada comando que será executado nas 4 fases (e os passos de coleta/consolidação).

### 7.4 Smoke test (1 modelo, 1 fase) — recomendado antes do run completo

Valida que a infra (container, GPU, dataset, pesos) está saudável antes de comprometer várias horas de GPU:

```bash
GPU_DEVICE_IDS="0,1"

docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" -it --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TORCH_HOME=/workspace/cache/torch \
    -e HOME=/workspace/cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GPU_DEVICE_IDS="${GPU_DEVICE_IDS}" \
    -e PIPELINE_NAME="smoke_test" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
    -v "$(pwd)/utils:/workspace/utils" \
    -v "$(pwd)/cache:/workspace/cache" \
    -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    yolo26_ft \
    bash /workspace/run_pipeline.sh \
        --models "n" \
        --phases "1" \
        --force
```

Se aparecer `weights/best.pt` em `logs/smoke_test/phase1_baseline/yolo26_nano_baseline/`, está pronto.

### 7.5 Execução completa (recomendado em `screen`/`tmux` no servidor)

#### Passo 1 — criar a sessão `screen`
```bash
screen -S yolo26_pipeline
```

#### Passo 2 — executar o pipeline completo dentro da sessão
```bash
GPU_DEVICE_IDS="0,1"
PIPELINE_NAME="pipeline_e2e_v1"   # mude para v2/v3/... a cada nova ablação

docker run --gpus "\"device=${GPU_DEVICE_IDS}\"" -it --rm \
    --ipc=host \
    --user "$(id -u):$(id -g)" \
    -e TORCH_HOME=/workspace/cache/torch \
    -e HOME=/workspace/cache \
    -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    -e GPU_DEVICE_IDS="${GPU_DEVICE_IDS}" \
    -e PIPELINE_NAME="${PIPELINE_NAME}" \
    -v "$(pwd)/datasets:/workspace/datasets" \
    -v "$(pwd)/logs:/workspace/logs" \
    -v "$(pwd)/yolo26_seg:/workspace/yolo26_seg" \
    -v "$(pwd)/utils:/workspace/utils" \
    -v "$(pwd)/cache:/workspace/cache" \
    -v "$(pwd)/run_pipeline.sh:/workspace/run_pipeline.sh:ro" \
    -v /etc/passwd:/etc/passwd:ro \
    -v /etc/group:/etc/group:ro \
    yolo26_ft \
    bash /workspace/run_pipeline.sh \
    2>&1 | tee "logs/${PIPELINE_NAME}_$(date -u +%Y%m%dT%H%M%SZ).log"
```

#### Passo 3 — destacar a sessão para o pipeline continuar rodando
Pressione **`Ctrl+A`** seguido de **`D`**. O pipeline continua executando em background.

#### Passo 4 — reanexar quando quiser ver o progresso
```bash
screen -r yolo26_pipeline
```

#### Passo 5 — monitorar de outro terminal
```bash
# Log mestre do orquestrador
tail -f logs/pipeline_e2e_v1/pipeline_runs/*/pipeline.log

# Log da fase atual (exemplo: Fase 4)
tail -f logs/pipeline_e2e_v1/pipeline_runs/*/phase4.log

# GPUs
watch -n 2 nvidia-smi
```

### 7.6 Execução em background sem `screen` (alternativa com `nohup`)
```bash
nohup bash run_pipeline.sh > "logs/pipeline_e2e_v1_$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 &
echo $! > pipeline.pid
```

### 7.7 Execuções parciais e re-execuções

| Cenário | Comando |
|---|---|
| Rodar só baseline para todos os modelos | `bash run_pipeline.sh --phases "1"` |
| Rodar só HPO + Otimizado + CV | `bash run_pipeline.sh --phases "2 3 4"` |
| Rodar tudo só para nano e small | `bash run_pipeline.sh --models "n s"` |
| Forçar re-execução completa (ignora artefatos existentes) | `bash run_pipeline.sh --force` |
| Nova ablação isolada em pasta separada | `bash run_pipeline.sh --pipeline-name pipeline_e2e_v2` |
| Custom search space ou mais iterações | `HPO_ITERATIONS=50 HPO_EPOCHS_PER_TRIAL=40 bash run_pipeline.sh --phases "2"` |

### 7.8 Variáveis de ambiente disponíveis

| Variável | Default | Uso |
|---|---|---|
| `DATA_YAML` | `/workspace/datasets/isic_2018_task1_yolo26/data.yaml` | Caminho do data.yaml. |
| `LOGS_ROOT` | `/workspace/logs` | Pai de todos os runs. |
| `PIPELINE_NAME` | `pipeline_e2e_v1` | Pasta isolada deste run. |
| `PROJECT` | `${LOGS_ROOT}/${PIPELINE_NAME}` | Override total (bypass composição). |
| `GPU_DEVICE_IDS` | `0,1` | GPUs (vírgula para DDP). |
| `P1_EPOCHS` / `P1_PATIENCE` | `120` / `20` | Fase 1. |
| `HPO_SPACE` / `HPO_ITERATIONS` / `HPO_EPOCHS_PER_TRIAL` / `HPO_PATIENCE` | `refined` / `30` / `30` / `10` | Fase 2. |
| `CV_K_FOLDS` / `CV_SEED` / `CV_EPOCHS` / `CV_PATIENCE` | `5` / `0` / `120` / `25` | Fase 4. |

### 7.9 Recuperando resultados do servidor remoto
```bash
rsync -avz --progress \
    -e "ssh -p 13508" \
    antoniovinicius@164.41.75.221:/home/antoniovinicius/projects/SANDBOX_YOLO26/logs/pipeline_e2e_v1/ \
    /home/avmoura_linux/Documents/unb/SANDBOX_YOLO26/logs/pipeline_e2e_v1/
```

---

## 8. O que reportar no artigo

### 8.1 Tabela principal (proveniente de `pipeline_summary/`)
A partir dos 3 arquivos consolidados, monte uma tabela com 5 linhas (uma por arquitetura) e as seguintes colunas:

| Modelo | mAP@50 (Baseline) | mAP@50 (Optimized) | mAP@50 (CV: mean ± std) | mAP@50-95 (Baseline / Opt / CV) | Precision (CV) | Recall (CV) | F1 (CV) |
|---|---|---|---|---|---|---|---|

Use as colunas `_m` (máscara) como métricas principais — segmentação. Reporte `_b` (bounding box) como suplementar.

### 8.2 Reprodutibilidade (seção "Methods")
Mencione explicitamente:
1. Ultralytics YOLO26 versão X (verificar `pip freeze` dentro do container).
2. Pesos COCO pré-treinados.
3. Split determinístico: `numpy.random.RandomState(seed=0)` ↔ `sklearn.KFold(shuffle=True, random_state=0)`.
4. Todos os treinos com `deterministic=True`, `seed=0`.
5. Hardware (GPU model, DDP `0,1`), `imgsz=640`, `batch=16` (treino) / `32` (HPO), epochs/patience por fase.
6. Otimizador `MuSGD`, cosine LR, AMP `False` no treino final (`True` durante HPO para velocidade).

### 8.3 Espaço de busca da HPO (seção "Hyperparameter Optimization")
Reporte a tabela `SEARCH_SPACE_REFINED` em `yolo26_seg/tune_all_models_v2.py` (linhas 115-140) — 15 hiperparâmetros, intervalos exatos. Justifique o uso do algoritmo genético embutido do Ultralytics e o orçamento (30 iter × 30 ep/trial).

### 8.4 Figuras sugeridas
- **Fitness curve** (`tune_fitness.png` por modelo, da Fase 2).
- **PR curve** dos modelos otimizados (Fase 3).
- **Boxplot** das métricas por fold (Fase 4 — usar `cv_consolidated.json` → `per_fold`).
- **Confusion matrix** (auto-gerada por Ultralytics).

---

## 9. Resumo executivo

* **4 arquivos novos** (`run_pipeline.sh`, `yolo26_seg/train_baseline_models.py`, `yolo26_seg/collect_phase_metrics.py`, `yolo26_seg/consolidate_cv_results.py`) e **0 modificações** nos 3 scripts pré-existentes — eles foram **reusados via flag `--project`**.
* **1 comando** roda o pipeline inteiro: `bash run_pipeline.sh` (dentro do container, com GPUs).
* **Idempotente** — pode interromper e retomar quando quiser.
* **Isolado** — `logs/pipeline_e2e_v1/` não toca em runs anteriores. Para ablações: `--pipeline-name pipeline_e2e_v2`.
* **Outputs prontos para o artigo** em `logs/pipeline_e2e_v1/pipeline_summary/{baseline,optimized}_metrics.{csv,json}` + `cv_consolidated.{csv,json}`.
