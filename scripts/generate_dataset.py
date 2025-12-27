#!/usr/bin/env python3
"""Generate Hugging Face ready artifacts for the ARC-AGI-2 dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

from datasets import (
    Dataset,
    DatasetDict,
    DatasetInfo,
    Features,
    Image,
    Sequence as HFSequence,
    SplitDict,
    SplitInfo,
    Value,
)
from PIL import Image as PilImage, ImageDraw, ImageFont
from tqdm import tqdm

# Official ARC palette approximations.
ARC_COLOR_MAP = {
    0: (0, 0, 0),  # black
    1: (0, 113, 188),  # blue
    2: (216, 82, 24),  # red
    3: (236, 176, 32),  # green/yellow-ish
    4: (125, 46, 141),  # purple
    5: (118, 171, 47),  # green
    6: (76, 189, 237),  # cyan
    7: (161, 19, 46),  # magenta
    8: (76, 76, 76),  # gray
    9: (153, 153, 51),  # olive
}

SYSTEM_PROMPT = (
    "You are an expert ARC-AGI problem solver. Given a few-shot set of input/output grids "
    "followed by a new input grid, you must produce the exact output grid in JSON array form."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", default="data", type=Path, help="Directory that contains training and evaluation folders.")
    parser.add_argument("--output-dir", default=Path("artifacts/hf-dataset"), type=Path, help="Where to write Hugging Face artifacts.")
    parser.add_argument("--max-image-size", type=int, default=200, help="Maximum width/height in pixels for rendered PNG grids.")
    parser.add_argument("--overwrite", action="store_true", help="Allow removing an existing output directory.")
    parser.add_argument(
        "--max-tasks-per-split",
        type=int,
        default=None,
        help="Optionally limit how many task JSON files to parse per split (useful for previews).",
    )
    parser.add_argument("--repo-id", type=str, default=None, help="Optional Hugging Face dataset repo id to push to (e.g. user/dataset).")
    parser.add_argument("--hf-token", type=str, default=None, help="Token to use when pushing to the Hugging Face Hub.")
    return parser.parse_args()


@dataclass
class TaskRecord:
    task_id: str
    split: str
    train_pairs: Sequence[Dict[str, List[List[int]]]]
    test_pairs: Sequence[Dict[str, List[List[int]]]]


def grid_to_text(grid: Sequence[Sequence[int]]) -> str:
    if not grid:
        return "<empty>"
    return "\n".join(" ".join(str(cell) for cell in row) for row in grid)


def grid_to_json_text(grid: Sequence[Sequence[int]]) -> str:
    return json.dumps(grid)


def stable_hash(payload: Sequence[Sequence[int]]) -> str:
    data = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def compute_cell_size(height: int, width: int, max_image_size: int) -> int:
    if height <= 0 or width <= 0:
        return max_image_size
    max_dim = max(height, width)
    return max(1, max_image_size // max_dim)


_FONT_CACHE: Dict[int, ImageFont.ImageFont] = {}


def _get_font(cell_size: int) -> ImageFont.ImageFont:
    size = max(8, cell_size // 2)
    if size in _FONT_CACHE:
        return _FONT_CACHE[size]
    try:
        font = ImageFont.truetype("DejaVuSansMono.ttf", size)
    except OSError:
        font = ImageFont.load_default()
    _FONT_CACHE[size] = font
    return font


def _text_color(background: Sequence[int]) -> tuple[int, int, int]:
    r, g, b = background
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return (0, 0, 0) if luminance > 186 else (255, 255, 255)


def grid_to_png(grid: Sequence[Sequence[int]], path: Path, max_image_size: int, annotate: bool) -> None:
    ensure_dir(path.parent)
    height = len(grid)
    width = len(grid[0]) if grid else 0
    cell_size = compute_cell_size(height or 1, width or 1, max_image_size)
    img_height = max(1, height) * cell_size
    img_width = max(1, width) * cell_size
    image = PilImage.new("RGB", (img_width, img_height), ARC_COLOR_MAP[0])
    draw = ImageDraw.Draw(image)
    font = _get_font(cell_size)
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            color = ARC_COLOR_MAP.get(value, (255, 255, 255))
            x0 = x * cell_size
            y0 = y * cell_size
            draw.rectangle((x0, y0, x0 + cell_size, y0 + cell_size), fill=color)
            if annotate:
                text = str(value)
                text_color = _text_color(color)
                bbox = draw.textbbox((0, 0), text, font=font)
                text_width = bbox[2] - bbox[0]
                text_height = bbox[3] - bbox[1]
                text_x = x0 + (cell_size - text_width) / 2
                text_y = y0 + (cell_size - text_height) / 2
                draw.text((text_x, text_y), text, fill=text_color, font=font)
    image.save(path)


def render_grid_versions(grid: Sequence[Sequence[int]], base_dir: Path, name: str, max_image_size: int) -> Dict[str, Path]:
    ensure_dir(base_dir)
    color_path = base_dir / f"{name}_color.png"
    annotated_path = base_dir / f"{name}_annotated.png"
    grid_to_png(grid, color_path, max_image_size, annotate=False)
    grid_to_png(grid, annotated_path, max_image_size, annotate=True)
    return {"color": color_path, "annotated": annotated_path}


def load_records(raw_root: Path, output_dir: Path, max_image_size: int, max_tasks_per_split: int | None) -> Dict[str, List[Dict]]:
    splits = {"train": raw_root / "training", "evaluation": raw_root / "evaluation"}
    results: Dict[str, List[Dict]] = {"train": [], "evaluation": []}
    for split_name, split_path in splits.items():
        task_paths = sorted(split_path.glob("*.json"))
        limited_paths = task_paths
        if max_tasks_per_split is not None:
            limited_paths = task_paths[:max_tasks_per_split]
        for task_path in tqdm(limited_paths, desc=f"Loading {split_name} tasks"):
            with task_path.open("r") as fp:
                payload = json.load(fp)
            task_id = task_path.stem
            train_pairs = payload.get("train", [])
            test_pairs = payload.get("test", [])
            task_image_dir = output_dir / "images" / split_name / task_id
            train_input_color_images: List[str] = []
            train_input_annotated_images: List[str] = []
            train_output_color_images: List[str] = []
            train_output_annotated_images: List[str] = []
            test_input_color_images: List[str] = []
            test_input_annotated_images: List[str] = []
            test_output_color_images: List[str] = []
            test_output_annotated_images: List[str] = []
            test_prompts: List[str] = []
            test_targets: List[str] = []
            test_input_texts: List[str] = []
            test_output_texts: List[str] = []
            test_conversations: List[Dict[str, List[str]]] = []
            for demo_index, demo_pair in enumerate(train_pairs):
                demo_dir = task_image_dir / "train"
                input_paths = render_grid_versions(demo_pair["input"], demo_dir, f"demo{demo_index}_input", max_image_size)
                output_paths = render_grid_versions(demo_pair["output"], demo_dir, f"demo{demo_index}_output", max_image_size)
                train_input_color_images.append(str(input_paths["color"]))
                train_input_annotated_images.append(str(input_paths["annotated"]))
                train_output_color_images.append(str(output_paths["color"]))
                train_output_annotated_images.append(str(output_paths["annotated"]))
            for test_index, test_pair in enumerate(test_pairs):
                test_dir = task_image_dir / "test"
                input_paths = render_grid_versions(test_pair["input"], test_dir, f"test{test_index}_input", max_image_size)
                output_paths = render_grid_versions(test_pair["output"], test_dir, f"test{test_index}_output", max_image_size)
                test_input_color_images.append(str(input_paths["color"]))
                test_input_annotated_images.append(str(input_paths["annotated"]))
                test_output_color_images.append(str(output_paths["color"]))
                test_output_annotated_images.append(str(output_paths["annotated"]))
                fewshot_prompt_lines = [
                    f"Task ID: {task_id}",
                    "You are given demonstration input/output grid pairs.",
                ]
                for idx, demo_pair in enumerate(train_pairs, start=1):
                    fewshot_prompt_lines.append(f"Example {idx} Input:\n{grid_to_text(demo_pair['input'])}")
                    fewshot_prompt_lines.append(f"Example {idx} Output:\n{grid_to_text(demo_pair['output'])}")
                fewshot_prompt_lines.append("Now solve the test input:")
                fewshot_prompt_lines.append(grid_to_text(test_pair["input"]))
                fewshot_prompt_lines.append("Respond with a JSON array-of-arrays.")
                fewshot_prompt = "\n\n".join(fewshot_prompt_lines)
                target = grid_to_json_text(test_pair["output"])
                test_prompts.append(fewshot_prompt)
                test_targets.append(target)
                test_input_texts.append(grid_to_text(test_pair["input"]))
                test_output_texts.append(grid_to_text(test_pair["output"]))
                test_conversations.append(
                    {
                        "role": ["system", "user", "assistant"],
                        "content": [SYSTEM_PROMPT, fewshot_prompt, target],
                    }
                )
            record = {
                "id": task_id,
                "task_id": task_id,
                "split": split_name,
                "train_pair_count": len(train_pairs),
                "test_pair_count": len(test_pairs),
                "train": [{"input": pair["input"], "output": pair["output"]} for pair in train_pairs],
                "test": [{"input": pair["input"], "output": pair["output"]} for pair in test_pairs],
                "test_outputs": [pair["output"] for pair in test_pairs],
                "train_input_image_color": train_input_color_images,
                "train_input_image_annotated": train_input_annotated_images,
                "train_output_image_color": train_output_color_images,
                "train_output_image_annotated": train_output_annotated_images,
                "test_input_image_color": test_input_color_images,
                "test_input_image_annotated": test_input_annotated_images,
                "test_output_image_color": test_output_color_images,
                "test_output_image_annotated": test_output_annotated_images,
                "test_input_texts": test_input_texts,
                "test_output_texts": test_output_texts,
                "test_prompts": test_prompts,
                "test_targets": test_targets,
                "test_conversations": test_conversations,
            }
            results[split_name].append(record)
    return results


def build_features() -> Features:
    grid_feature = HFSequence(HFSequence(Value("int32")))
    conversation_turn_feature = Features(
        {
            "role": HFSequence(Value("string")),
            "content": HFSequence(Value("string")),
        }
    )
    conversation_feature = HFSequence(conversation_turn_feature)
    image_list_feature = HFSequence(Image())
    return Features(
        {
            "id": Value("string"),
            "task_id": Value("string"),
            "split": Value("string"),
            "train_pair_count": Value("int32"),
            "test_pair_count": Value("int32"),
            "train": HFSequence({"input": grid_feature, "output": grid_feature}),
            "test": HFSequence({"input": grid_feature, "output": grid_feature}),
            "test_outputs": HFSequence(grid_feature),
            "train_input_image_color": image_list_feature,
            "train_input_image_annotated": image_list_feature,
            "train_output_image_color": image_list_feature,
            "train_output_image_annotated": image_list_feature,
            "test_input_image_color": image_list_feature,
            "test_input_image_annotated": image_list_feature,
            "test_output_image_color": image_list_feature,
            "test_output_image_annotated": image_list_feature,
            "test_input_texts": HFSequence(Value("string")),
            "test_output_texts": HFSequence(Value("string")),
            "test_prompts": HFSequence(Value("string")),
            "test_targets": HFSequence(Value("string")),
            "test_conversations": conversation_feature,
        }
    )


def write_parquet(datasets_by_split: Dict[str, Dataset], data_dir: Path) -> Dict[str, int]:
    ensure_dir(data_dir)
    split_sizes: Dict[str, int] = {}
    for split_name, dataset in datasets_by_split.items():
        target_path = data_dir / f"{split_name}-00000-of-00001.parquet"
        dataset.to_parquet(target_path)
        split_sizes[split_name] = target_path.stat().st_size
    return split_sizes


def write_preview_jsonl(records: Dict[str, List[Dict]], output_dir: Path) -> None:
    preview_dir = output_dir / "preview"
    ensure_dir(preview_dir)
    for split_name, rows in records.items():
        target_path = preview_dir / f"{split_name}.jsonl"
        with target_path.open("w") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")


def build_dataset_info(features: Features, split_sizes: Dict[str, int], datasets_by_split: Dict[str, Dataset], output_dir: Path) -> None:
    split_dict = SplitDict()
    for split_name, dataset in datasets_by_split.items():
        split_dict.add(SplitInfo(name=split_name, num_examples=len(dataset), num_bytes=split_sizes[split_name], dataset_name="arc_agi_2"))
    image_bytes = 0
    images_dir = output_dir / "images"
    if images_dir.exists():
        for path in images_dir.rglob("*.png"):
            image_bytes += path.stat().st_size
    info = DatasetInfo(
        description="ARC-AGI-2 reformatted for language-first and multimodal training.",
        citation="""@misc{arcagi2, title={Abstraction and Reasoning Corpus for AGI v2}, year={2024}, author={ARC Prize Team}}""",
        homepage="https://arcprize.org/",
        license="apache-2.0",
        version="2.0.0",
        features=features,
        builder_name="arc_agi_2",
        config_name="default",
        splits=split_dict,
        supervised_keys=None,
    )
    info.dataset_size = sum(split_sizes.values())
    info.size_in_bytes = info.dataset_size + image_bytes
    info.write_to_directory(output_dir)


def maybe_push_to_hub(datasets_by_split: Dict[str, Dataset], repo_id: str, token: str | None) -> None:
    if not repo_id:
        return
    dataset_dict = DatasetDict(datasets_by_split)
    dataset_dict.push_to_hub(repo_id, token=token, private=False)


def main() -> None:
    args = parse_args()
    output_dir: Path = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"{output_dir} already exists. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)
    records = load_records(args.raw_root, output_dir, args.max_image_size, args.max_tasks_per_split)
    write_preview_jsonl(records, output_dir)
    features = build_features()
    datasets_by_split: Dict[str, Dataset] = {}
    for split_name, rows in records.items():
        datasets_by_split[split_name] = Dataset.from_list(rows, features=features)
    split_sizes = write_parquet(datasets_by_split, output_dir / "data")
    build_dataset_info(features, split_sizes, datasets_by_split, output_dir)
    maybe_push_to_hub(datasets_by_split, args.repo_id, args.hf_token)
    print(f"Wrote Hugging Face artifacts to {output_dir}")


if __name__ == "__main__":
    main()
