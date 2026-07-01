# LettuceDetect — Training

`train.py` is the actual training script used to fine-tune the ModernBERT
(LettuceDetect) hallucination detector, taken as-is from the LettuceDetect
library (`LettuceDetect/scripts/train.py`).

No data is bundled here — supply your own RAGTruth-format training JSON (see
`../../hallucination_generation_pipeline/` for how to produce one from this
project's generated hallucinations via `make_lettucedetect_data.py`).

## 1. Install deps

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install lettucedetect torch transformers
```

## 2. Data format

`train.py` expects a JSON file matching `HallucinationData`/`HallucinationSample`
(from `lettucedetect.datasets.hallucination_dataset`) — each sample has
`prompt`, `answer`, `labels` (`[{start, end, label}]`), and a `split` field
(`"train"` samples are used; a 10% dev split is carved out automatically).

## 3. Train

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --ragtruth-path train.json \
  --model-name answerdotai/ModernBERT-base \
  --output-dir results_hallucination_detector \
  --batch-size 1 \
  --epochs 6 \
  --learning-rate 1e-5 \
  --grad-accum 8
```

### Arguments

| Argument | Default | Description |
| --- | --- | --- |
| `--ragtruth-path` | `data/ragtruth/ragtruth_data.json` | Path to the training data JSON |
| `--ragbench-path` | *(optional)* | Additional RAGBench training data; if omitted, only the ragtruth-path data is used |
| `--model-name` | `answerdotai/ModernBERT-base` | Pretrained model name or local path |
| `--output-dir` | `output/hallucination_detector` | Where the trained model is saved |
| `--batch-size` | `4` | Per-device batch size |
| `--epochs` | `6` | Training epochs |
| `--learning-rate` | `1e-5` | Learning rate |
| `--grad-accum` | `8` | Gradient accumulation steps (effective batch = batch-size × grad-accum) |

- A 10% dev split is automatically carved out of the training data for validation.
- Random seed is fixed at `123` for reproducibility.

## 4. Then evaluate / run inference

See `../inference/README.md` — point `evaluate_save.py --model` at
`--output-dir` from the training run above.
