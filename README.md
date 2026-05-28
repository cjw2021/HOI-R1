# HOI-R1: Exploring the Potential of Multimodal Large Language Models for Human-Object Interaction Detection

[![arXiv](https://img.shields.io/badge/arXiv-2510.05609-b31b1b.svg)](https://arxiv.org/abs/2510.05609)
[![Hugging Face](https://img.shields.io/badge/🤗%20HuggingFace-Collection-yellow)](https://huggingface.co/collections/thxplz/hoi-r1)

This repository contains the official resources for **HOI-R1**, a research project that explores the potential of **Multimodal Large Language Models (MLLMs)** for **Human-Object Interaction (HOI) Detection**.

HOI-R1 is inspired by recent advances in reinforcement learning for large language models and investigates how vision-language models can reason about and detect human-object interactions more effectively.

---

## 🔍 Overview

- **Task**: Human-Object Interaction Detection (HOID)
- **Our Motivation**:  
  Leverage the reasoning capability of Multimodal LLMs and reinforcement learning–style optimization to explore HOI detection performance.

![hoi-r1-arch](https://cdn-uploads.huggingface.co/production/uploads/63119ce2fb65b9a3e2f75e3c/tHYWwrnqBAHsoo8lIOtnM.jpeg)

---

## 🤗 Model Weights

All model checkpoints are collected on Hugging Face:

👉 **HOI-R1 Collection**  
https://huggingface.co/collections/thxplz/hoi-r1

Currently released:

- **HOI-R1 (Qwen2.5-VL)** — weights public
- **HOI-R1 (Qwen3-VL)** — weights public
- **HOI-R1 (Rex-Omni)** — coming soon

---

## ⚙️ Evaluation

`eval_qwen3.py` runs HOI detection inference on the HICO-DET test set with a
Qwen3-VL model and saves predictions for later metric computation. It supports
multi-GPU inference (one process per GPU) and optional sharding of the dataset
into sections so the evaluation can be split across multiple machines/runs.

**Requirements**

```bash
pip install torch transformers pillow numpy tqdm
# flash-attention is used by default (attn_implementation="flash_attention_2")
pip install flash-attn --no-build-isolation
```

**Data**

Download the [HICO-DET](https://websites.umass.edu/hoi/datasets/) dataset. You
need the `test2015` images and a `test_hico.json` annotation file.

**Run**

```bash
python eval_qwen3.py \
    --model_path thxplz/HOI-R1_Qwen2.5-VL-3B-Instruct \
    --test_anno /path/to/HICO_Det/annotations/test_hico.json \
    --image_folder /path/to/HICO_Det/images/test2015 \
    --output_folder ./evaluate_predictions/hoi-r1 \
    --num_gpus 8 --section 0 --max_section 1
```

**Arguments**

| Argument | Description |
| --- | --- |
| `--model_path` | Model checkpoint or HF repo id (default: `Qwen/Qwen3-VL-4B-Instruct`). |
| `--test_anno` | Path to the HICO-DET `test_hico.json` annotation file. |
| `--image_folder` | Path to the HICO-DET `test2015` image folder. |
| `--output_folder` | Folder to save prediction JSON files. |
| `--num_gpus` | Number of GPUs / parallel inference processes. |
| `--section` / `--max_section` | Split the dataset into `max_section` shards and process shard `section` (0-based). Use `--max_section 1` to run on the full set. |
| `--existing_merged_json` | Optional merged prediction JSON; already-processed images are skipped (resume support). |

Predictions are written as `pred_<task_id>_<timestamp>.json` files containing
the raw model `outputs`, parsed `preds`, and corresponding `gt_paths`.
Post-processing of the raw model output is handled by `postprocess.py`.

**Merge predictions**

Since inference is split across GPUs (and optionally sections), the per-process
prediction files need to be merged into a single `merged_output.json` before
metric computation:

```bash
python merge_results.py --folder_path ./evaluate_predictions/hoi-r1
```

This recursively gathers every `pred_*.json` under `--folder_path`, removes
duplicate images, and writes `merged_output.json` into that folder. Pass `--re`
to re-run post-processing on the stored raw `outputs` instead of reusing the
already-parsed `preds`. The merged file can also be fed back to
`eval_qwen3.py` via `--existing_merged_json` to resume an interrupted run.

---

## 🛠 TODO

- [ ] Add qualitative visualization examples
- [x] Release inference & evaluation scripts
- [ ] Add dataset preprocessing pipeline
- [ ] Provide detailed training configuration and hyperparameters  
- [ ] Release training code
- [ ] Release additional model variants

---

## 📌 Citation

If you find this work useful, please consider citing:

```bibtex
@article{chen2025hoi,
  title={HOI-R1: Exploring the Potential of Multimodal Large Language Models for Human-Object Interaction Detection},
  author={Chen, Junwen and Xiong, Peilin and Yanai, Keiji},
  journal={arXiv preprint arXiv:2510.05609},
  year={2025}
}
