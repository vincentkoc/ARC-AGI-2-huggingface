# Abstraction and Reasoning Corpus for Artificial General Intelligence v2 (ARC-AGI-2)

This repository contains a Hugging Face–ready version of the ARC-AGI-2 task data (ARC-AGI-1 can be found [here](https://github.com/fchollet/arc-agi)). In addition to mirroring the canonicalized JSON files, it ships with additional pre-processing that:

- keeps the original train/evaluation splits
- emits Parquet shards so the dataset supports `datasets` streaming
- renders every grid into PNG form for multimodal research
- derives few-shot prompts, chat conversations, and text fields (`task_id`, `input`, `output`, etc.)
- computes auxiliary metadata such as grid sizes and SHA-256 hashes

Use this repo when you want to reproduce or customize the Hugging Face dataset card (`vincentkoc/arc-agi-2`) or build new derived assets. The canonical GitHub location is [https://github.com/vincentkoc/ARC-AGI-2-huggingface](https://github.com/vincentkoc/ARC-AGI-2-huggingface).

*"ARC can be seen as a general artificial intelligence benchmark, as a program synthesis benchmark, or as a psychometric intelligence test. It is targeted at both humans and artificially intelligent systems that aim at emulating a human-like form of general fluid intelligence."*

A foundational description of the dataset, its goals, and its underlying logic, can be found in: [On the Measure of Intelligence](https://arxiv.org/abs/1911.01547), the [ARC-AGI-2 Presentation](https://docs.google.com/presentation/d/1hQrGh5YI6MK3PalQYSQs4CQERrYBQZue8PBLjjHIMgI/edit?usp=sharing) and [ARC-AGI-2 Technical Report](http://arcprize.org/blog/arc-agi-2-technical-report)

## Dataset composition

ARC-AGI-2 contains 1,000 public training tasks and 120 public evaluation tasks.

The training tasks are intended to demonstrate the task format and the Core Knowledge priors used by ARC-AGI. They can be used for training AI models.
The public evaluation tasks are intended for testing AI models that have never seen these tasks before. Average human performance on these tasks in our test sample was 66%.

ARC-AGI-2 also features two private test sets not included in the repo:

- A semi-private set intended for testing remotely-hosted commercial models with low leakage probability. It is calibrated to be the same human-facing difficulty as the public evaluation set.
- A fully-private set intended for testing self-contained models during the ARC Prize competition, with near-zeo leakage probability. It is also calibrated to be the same difficulty.

This multi-tiered structure allows for both open research and a secure, high-stakes competition.

## Task success criterion

A test-taker is said to solve a task when, upon seeing the task for the first time, they are able to produce the correct output grid for *all* test inputs in the task (this includes picking the dimensions of the output grid). For each test input, the test-taker is allowed 2 trials (this holds for all test-takers, either humans or AI).

## Task file format

The `data` directory contains two subdirectories:

- `data/training`: contains the task files for training (1000 tasks). Use these to prototype your algorithm or to train your algorithm to acquire ARC-relevant cognitive priors. This set combines tasks from ARC-AGI-1 as well as new tasks.
- `data/evaluation`: contains the task files for evaluation (120 tasks). Use these to evaluate your final algorithm. To ensure fair evaluation results, do not leak information from the evaluation set into your algorithm (e.g. by looking at the evaluation tasks yourself during development, or by repeatedly modifying an algorithm while using its evaluation score as feedback). Each task in `evaluation` has been solved by a minimum of 2 people (many tasks were solved by more) in 2 attempts or less in a controlled test.

The tasks are stored in JSON format. Each task JSON file contains a dictionary with two fields:

- `"train"`: demonstration input/output pairs. It is a list of "pairs" (typically 3 pairs).
- `"test"`: test input/output pairs. It is a list of "pairs" (typically 1-2 pair).

A "pair" is a dictionary with two fields:

- `"input"`: the input "grid" for the pair.
- `"output"`: the output "grid" for the pair.

A "grid" is a rectangular matrix (list of lists) of integers between 0 and 9 (inclusive). The smallest possible grid size is 1x1 and the largest is 30x30.

When looking at a task, a test-taker has access to inputs & outputs of the demonstration pairs, plus the input(s) of the test pair(s). The goal is to construct the output grid(s) corresponding to the test input grid(s), using 3 trials for each test input. "Constructing the output grid" involves picking the height and width of the output grid, then filling each cell in the grid with a symbol (integer between 0 and 9, which are visualized as colors). Only *exact* solutions (all cells match the expected answer) can be said to be correct.

## Hugging Face dataset schema

The `scripts/generate_dataset.py` utility repackages the raw JSON files into a single Hugging Face dataset that matches the ARC Prize loader expectations:

- **Splits:** `train` (≈1000 tasks) and `evaluation` (≈120 tasks). Each row equals one task JSON.
- **Required fields per row:**
  - `task_id`: hex stem of the source JSON filename.
  - `train`: list of `{ "input": [[int]], "output": [[int]] }`.
  - `test`: list of `{ "input": [[int]], "output": [[int]] }` (public tasks include solutions).
  - `test_outputs`: duplicate of `test[*].output` for convenience.
- **Extra fields:** grouped `Image` columns for every rendered grid (`train_input_image_color`, `train_input_image_annotated`, `train_output_image_color`, etc.), plus `test_input_texts`, `test_output_texts`, `test_prompts`, `test_targets`, and `test_conversations` (chat-style traces). All grids stay as ragged lists-of-lists with integers 0–9; no padding or shuffling is applied.

This shape allows downstream consumers to simply call `load_dataset("vincentkoc/arc_agi_2", split="train")` and index per-task records while still having access to multimodal renderings and LLM-ready prompts on each row.


## Usage of the testing interface

You can view tasks on [ARCPrize.org/play](https://arcprize.org/play) or clone the [ARC-AGI-1 testing interface](https://github.com/fchollet/ARC-AGI/tree/master/apps). Open it in a web browser (Chrome recommended). It will prompt you to select a task JSON file.

After loading a task, you will enter the test space, which looks like this:

![test space](https://arc-benchmark.s3.amazonaws.com/figs/arc_test_space.png)

On the left, you will see the input/output pairs demonstrating the nature of the task. In the middle, you will see the current test input grid. On the right, you will see the controls you can use to construct the corresponding output grid.

You have access to the following tools:

### Grid controls

- Resize: input a grid size (e.g. "10x20" or "4x4") and click "Resize". This preserves existing grid content (in the top left corner).
- Copy from input: copy the input grid to the output grid. This is useful for tasks where the output consists of some modification of the input.
- Reset grid: fill the grid with 0s.

### Symbol controls

- Edit: select a color (symbol) from the color picking bar, then click on a cell to set its color.
- Select: click and drag on either the output grid or the input grid to select cells.
    - After selecting cells on the output grid, you can select a color from the color picking to set the color of the selected cells. This is useful to draw solid rectangles or lines.
    - After selecting cells on either the input grid or the output grid, you can press C to copy their content. After copying, you can select a cell on the output grid and press "V" to paste the copied content. You should select the cell in the top left corner of the zone you want to paste into.
- Floodfill: click on a cell from the output grid to color all connected cells to the selected color. "Connected cells" are contiguous cells with the same color.

### Answer validation

When your output grid is ready, click the green "Submit!" button to check your answer. We do not enforce the 2-trials rule.

After you've obtained the correct answer for the current test input grid, you can switch to the next test input grid for the task using the "Next test input" button (if there is any available; most tasks only have one test input).

When you're done with a task, use the "load task" button to open a new task.

## Publishing to Hugging Face

Use the utilities in `scripts/generate_dataset.py` to transform the raw ARC-AGI-2 JSON tasks into Parquet shards, PNG renderings, and LLM-ready conversation data. The script writes everything into the folder given by `--output-dir`, including:

- `images/`: per-task folders with both plain-color and digit-annotated PNGs for every demo/test grid
- `preview/*.jsonl`: human-readable rows for quick inspection
- `data/*.parquet`: Hugging Face–friendly shards for upload

Each demo/test grid gets two PNGs: a plain-color rendering plus an annotated rendering with the integer token centered in every cell. Files are grouped by split and task under `images/<split>/<task_id>/` for easy navigation.

The script never pushes to Hugging Face unless you explicitly supply `--repo-id`.

Run it locally for the full dataset:

```
pip install -r requirements.txt
python scripts/generate_dataset.py --raw-root data --output-dir artifacts/hf-dataset --overwrite
```

For a lightweight preview (e.g. first two tasks per split) that you can inspect before publishing:

```
python scripts/generate_dataset.py \
  --raw-root data \
  --output-dir artifacts/hf-preview \
  --max-tasks-per-split 2 \
  --overwrite
```

Open `artifacts/hf-preview/preview/train.jsonl` (and `evaluation.jsonl`) or the PNGs under `artifacts/hf-preview/images/...` to review the generated content.

See `README_hf.md` plus `.github/workflows/publish_hf.yml` for the automated release workflow that mirrors the generated artifacts to the Hugging Face Hub.
