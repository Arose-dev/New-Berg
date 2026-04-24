"""
LEGO dataset adapter for the spatial reasoning pipeline.

Downloads the LEGO-Puzzles benchmark TSV and converts it into the local
annotations.json format expected by the pipeline.

The LEGO benchmark has 1,100 multiple-choice questions across 11 spatial
reasoning categories.  Each question references one or more images that are
base64-encoded inside the TSV.

Usage:
    python -m datasets.lego_dataset              # LEGO-Lite (default, 400 Qs)
    python -m datasets.lego_dataset --full       # full benchmark (1100 Qs)
    python -m datasets.lego_dataset --max-questions 20  # small test subset
"""

import argparse
import base64
import io
import json
import os
import string
import sys

import pandas as pd
from PIL import Image

LEGO_TSV_URL = "https://opencompass.openxlab.space/utils/VLMEval/LEGO.tsv"
LEGO_TSV_MD5 = "d595f50e1fb4d4eb12cbc95297893ffc"

LEGO_CATEGORIES = [
    "adjacency", "backwards", "dependency", "height", "multi_view",
    "next_step", "ordering", "outlier", "position", "rotation",
    "rotation_status",
]

# Lite mode: 100 questions per category across these 4 categories (400 total)
LITE_CATEGORIES = ["height", "position", "rotation", "ordering"]
LITE_PER_CATEGORY = 100

SUPPORTED_TEXT_QUESTION_TYPES = {
    "mcq",
    "multiple-choice",
    "bool",
    "boolean",
    "true_false",
    "true-false",
}
IMAGE_OUTPUT_MARKERS = (
    "generate an image",
    "generated image",
    "generate the image",
    "output an image",
    "create an image",
    "draw the next step",
    "synthesize an image",
)


def download_lego_tsv(dest_path: str) -> str:
    """Download the LEGO TSV if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"LEGO TSV already exists at {dest_path}")
        return dest_path

    print(f"Downloading LEGO TSV from {LEGO_TSV_URL} ...")
    import urllib.request
    import ssl
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    with urllib.request.urlopen(LEGO_TSV_URL, context=ctx) as r, open(dest_path, "wb") as f:
        f.write(r.read())
    print(f"Saved to {dest_path}")
    return dest_path


def decode_base64_image(b64_str: str) -> Image.Image:
    """Decode a base64-encoded image string into a PIL Image."""
    image_data = base64.b64decode(b64_str)
    return Image.open(io.BytesIO(image_data)).convert("RGB")


def build_mcq_question_text(row: pd.Series) -> str:
    """
    Build the full question text including MCQ options.

    For supported text-only multiple-choice questions, asks the model to
    pick a single option letter.
    """
    question = row["question"]
    hint = row.get("hint", None)
    options = {}
    for letter in string.ascii_uppercase:
        if letter in row.index and pd.notna(row[letter]):
            options[letter] = row[letter]

    text = ""
    if pd.notna(hint) if isinstance(hint, str) else hint:
        text += f"Hint: {hint}\n"

    text += f"Question: {question}\n"

    if options:
        text += "Options:\n"
        for key, val in options.items():
            text += f"  {key}. {val}\n"

    text += (
        "Answer with only the letter of the correct option "
        "(for example: 'A', 'B', 'C', or 'D')."
    )

    return text


def is_text_only_choice_question(row: pd.Series) -> bool:
    """Keep only MCQ / true-false questions that do not depend on image outputs."""
    question_type = str(row.get("question_type", "mcq")).strip().lower()
    if question_type not in SUPPORTED_TEXT_QUESTION_TYPES:
        return False

    category = str(row.get("category", "")).strip().lower()
    if "generation" in category:
        return False

    question_bits = [str(row.get("question", "")), str(row.get("hint", ""))]
    for letter in string.ascii_uppercase:
        if letter in row.index and pd.notna(row[letter]):
            question_bits.append(str(row[letter]))
    combined = " ".join(question_bits).lower()

    return not any(marker in combined for marker in IMAGE_OUTPUT_MARKERS)


def convert_lego_to_benchmark_questions(
    tsv_path: str,
    output_dir: str,
    max_questions: int = -1,
    lite: bool = True,
) -> str:
    """
    Convert the LEGO TSV into annotations.json + extracted images.

    Returns the path to the annotations.json file.
    """
    print(f"Loading LEGO TSV from {tsv_path} ...")
    df = pd.read_csv(tsv_path, sep="\t")
    print(f"Loaded {len(df)} rows")

    if lite:
        lite_cats = set(LITE_CATEGORIES)
        df = df[df["category"].str.strip().str.lower().isin(lite_cats)]
        df = pd.concat([
            g.head(LITE_PER_CATEGORY)
            for _, g in df.groupby("category", sort=False)
        ]).reset_index(drop=True)
        print(
            f"Lite mode: up to {LITE_PER_CATEGORY} questions each from "
            f"{LITE_CATEGORIES} -> {len(df)} rows"
        )

    before = len(df)
    df = df[df.apply(is_text_only_choice_question, axis=1)]
    print(f"Filtered to text-only MCQ/TF questions: {before} -> {len(df)} rows")

    if max_questions > 0:
        df = df.head(max_questions)
        print(f"Limited to {len(df)} usable questions")

    images_dir = os.path.join(output_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    questions = []
    image_cache = {}

    for idx, row in df.iterrows():
        question_text = build_mcq_question_text(row)

        # Extract answer
        answer = str(row.get("answer", ""))

        # Extract category
        category = str(row.get("category", "unknown"))

        # Extract question type
        question_type = str(row.get("question_type", "mcq"))

        # Keep answers as short string labels for evaluation.
        answer_type = "str"

        # Handle images - LEGO TSV encodes images as base64 in the 'image' column
        # Some rows may have multiple images separated by special encoding
        image_filename = f"lego_{row['index']}.png"
        image_path = os.path.join(images_dir, image_filename)

        if not os.path.exists(image_path):
            try:
                if "image" in row.index and pd.notna(row["image"]):
                    img = decode_base64_image(row["image"])
                    img.save(image_path)
                elif "image_path" in row.index and pd.notna(row["image_path"]):
                    # Image paths might be listed - use the first one
                    img_path_str = str(row["image_path"])
                    if "|" in img_path_str:
                        img_path_str = img_path_str.split("|")[0]
                    if os.path.exists(img_path_str):
                        img = Image.open(img_path_str).convert("RGB")
                        img.save(image_path)
                    else:
                        print(f"Warning: image not found for index {row['index']}")
                        continue
            except Exception as e:
                print(f"Warning: failed to decode image for index {row['index']}: {e}")
                continue

        # Build pipeline question dict
        question_dict = {
            "image_index": str(row["index"]),
            "question_index": str(row["index"]),
            "image_filename": image_filename,
            "question": question_text,
            "answer": answer,
            "answer_type": answer_type,
            "category": category,
            "question_type": question_type,
        }
        questions.append(question_dict)

    annotations = {"questions": questions}
    annotations_path = os.path.join(output_dir, "annotations.json")
    with open(annotations_path, "w") as f:
        json.dump(annotations, f, indent=2)

    print(f"Created {len(questions)} questions in {annotations_path}")
    print(f"Images saved to {images_dir}")
    return annotations_path


def main():
    parser = argparse.ArgumentParser(description="Convert LEGO benchmark to pipeline format")
    parser.add_argument(
        "--tsv-path",
        default=None,
        help="Path to existing LEGO.tsv (will download if not provided)",
    )
    parser.add_argument(
        "--output-dir",
        default="data/lego",
        help="Output directory for annotations.json and images",
    )
    parser.add_argument(
        "--max-questions",
        default=-1,
        type=int,
        help="Maximum number of questions to convert (-1 for all)",
    )
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use LEGO-Lite subset (default)",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Use the full LEGO benchmark instead of LEGO-Lite",
    )
    args = parser.parse_args()

    if args.tsv_path is None:
        args.tsv_path = os.path.join(args.output_dir, "LEGO.tsv")
        download_lego_tsv(args.tsv_path)

    convert_lego_to_benchmark_questions(
        args.tsv_path,
        args.output_dir,
        max_questions=args.max_questions,
        lite=not args.full,
    )


if __name__ == "__main__":
    main()
