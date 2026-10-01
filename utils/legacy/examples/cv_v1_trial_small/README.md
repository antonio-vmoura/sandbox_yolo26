# Trial run — `train_all_models_cv.py` (small, cv_v1)

Esta pasta contém o resultado de uma **execução de validação (smoke test)** do
script `yolo26_seg/train_all_models_cv.py` no tamanho `small`, focada apenas em
provar que a divisão K-Fold e o registro de métricas funcionam ponta-a-ponta.

> **Importante:** o ISIC 2018 Task 1 e os pesos `yolo26*-seg.pt` não estavam
> disponíveis no ambiente onde a CI/teste rodou. Para o trial usamos o dataset
> `ph2` (que já está no repositório, 153 imagens, 1 classe — também
> segmentação de lesões cutâneas) e os pesos públicos `yolo11n-seg.pt` apenas
> para validar o pipeline. Em produção (Docker + GPUs) o script deve ser
> executado **sem** `--weights-override` e apontando para o `data.yaml` do
> ISIC.

## Arquivos

| Arquivo | Descrição |
|--------|-----------|
| `data_ph2.yaml` | `data.yaml` usado no trial (ponteiro para `datasets/ph2`). |
| `best_hyperparameters_stub_small.yaml` | HP stub colocado em `<project>/hpo/hpo_v3/tune_isic_2018_task_1_small/best_hyperparameters.yaml` para o trial. |
| `metrics_per_fold.csv` | Métricas por fold geradas pelo script. |
| `metrics_summary.json` | Agregação `mean ± std` por métrica. |

## Comando exato do trial

```bash
python yolo26_seg/train_all_models_cv.py \
    --models small \
    --data utils/examples/cv_v1_trial_small/data_ph2.yaml \
    --project /tmp/trial_workspace/logs \
    --device cpu \
    --epochs 1 \
    --patience 0 \
    --batch 2 \
    --imgsz 256 \
    --workers 0 \
    --weights-override yolo11n-seg.pt
```

> O HP stub precisa ser copiado para
> `/tmp/trial_workspace/logs/hpo/hpo_v3/tune_isic_2018_task_1_small/best_hyperparameters.yaml`
> antes da execução (mesmo nome do esperado em produção).

## O que o trial validou

1. **Divisão K-Fold determinística (seed=0):**
   - Pool: 153 imagens (train+val do `ph2`).
   - 5 folds, sem leakage (`train ∩ val = ∅`) e cobertura total
     (`train ∪ val = pool` em cada fold).
   - Reprodutível: mesma seed → mesmos splits.
2. **Geração de arquivos por fold:**
   - `splits/fold_<k>/{train.txt, val.txt, data.yaml}` corretamente criados.
   - `data.yaml` preserva `nc`/`names` do template e referencia paths absolutos.
3. **Registro de métricas:**
   - `results.csv` de cada fold parseado pela melhor época
     (`metrics/mAP50-95(M)`).
   - F1 derivado de precision/recall (Box e Mask).
   - Agregação `mean ± std` salva em CSV (por fold) e JSON (sumário).

Os números reportados não são significativos do ponto de vista de modelagem
(apenas 1 época em CPU com `yolo11n-seg`). Eles servem **somente** para
demonstrar que o pipeline lê, treina, valida e agrega corretamente.
