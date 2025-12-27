---
license: apache-2.0
tags:
  - arc
  - program-synthesis
  - reasoning
  - multimodal
task_categories:
  - question-answering
datasets:
  - arc-agi-2
---

# ARC-AGI-2 Few-Shot Conversations

ARC-AGI-2 is a benchmark of 1,000 public training tasks and 120 public evaluation tasks for assessing reasoning systems. This repository packages the public tasks into a Hugging Face–friendly format with:

- canonical/original arc-agi 2 train/evaluation splits
- Parquet shards for fast downloads & streaming
- per-example PNG renderings of every grid (demonstration and test)
- text prompts & full conversations ready for LLM fine-tuning

## Dataset structure

Each row corresponds to a *test* grid inside an ARC task. Demonstration (few-shot) pairs are stored alongside the test pair so that finetuning-ready prompts can be composed without extra processing.

| Column | Type | Description |
| ------ | ---- | ----------- |
| `id` | string | Unique identifier `<task_id>__test_<idx>` |
| `task_id` | string | Original ARC file stem |
| `split` | string | `train` (1,000 tasks) or `evaluation` (120 tasks) |
| `train` | list[{`input`, `output`}] | Demonstration grids (lists-of-lists of ints) |
| `test` | list[{`input`, `output`}] | Held-out grids (solutions included for public data) |
| `test_outputs` | list[list[list[int]]] | Convenience copy of `test[*].output` |
| `train_input_image_color` / `_annotated` | list[image] | PNGs for each demo input (plain palette + overlaid digits) |
| `train_output_image_color` / `_annotated` | list[image] | PNGs for each demo output |
| `test_input_image_color` / `_annotated` | list[image] | PNGs for each test input |
| `test_output_image_color` / `_annotated` | list[image] | PNGs for each test output (solutions) |
| `test_input_texts` / `test_output_texts` | list[str] | Plain-text renderings of each test pair |
| `test_prompts` / `test_targets` | list[str] | LLM-friendly prompts + JSON answers per test grid |
| `test_conversations` | dict | Nested `role`/`content` arrays for chat fine-tuning (one conversation per test) |

Images are rendered at up to 200×200 pixels with the canonical ARC palette, ensuring they display properly on the Hub and work with vision-language models.

## Usage

```python
from datasets import load_dataset

ds = load_dataset("vincentkoc/arc_agi_2_fewshot", split="train", streaming=True)
for row in ds.take(1):
    print(row["task_id"], row["fewshot_prompt"])
```

Indices align across the lists: `train[i]` corresponds to `train_input_image_color[i]`, `train_output_image_color[i]`, etc. To fine-tune an LLM with supervised signals, zip `test_prompts` with `test_targets` or use `test_conversations`. For multimodal agents, choose whichever variant you need from the image columns—every demo/test grid is available as both a pure color PNG and an annotated PNG with the numeric token rendered on top.

## Reproducing

```
pip install -r requirements.txt
python scripts/generate_dataset.py --raw-root data --output-dir artifacts/hf-dataset --overwrite
```

Set `--repo-id` and `--hf-token` to push directly to the Hugging Face Hub. The GitHub Action in this repo automates that process upon every release.
