"""
Run Qwen3-VL-MoE inference on the LEGO-Puzzles benchmark with expert activation logging.

Direct-inference alternative to run_lego.py.  Instead of the 4-step
program-synthesis pipeline, each question is fed directly to
Qwen3-VL-30B-A3B-Instruct and the MoE expert activations are logged
per token, per layer via forward hooks.

RunPod usage (env vars set in pod template or docker run -e):
    DATA_DIR      — where LEGO.tsv and images live  (default: /workspace/data/lego)
    RESULTS_DIR   — where results are written        (default: /workspace/results)
    HF_TOKEN      — HuggingFace token for gated models (optional)
    BENCHMARK_MODE — "lite" or "full"               (default: lite)
    MODEL_NAME    — HuggingFace model ID             (default: Qwen/Qwen3-VL-30B-A3B-Instruct)

CLI usage (flags override env vars):
    python run_lego_qwen.py                          # LEGO-Lite, 400 questions
    python run_lego_qwen.py --full                   # all 1100 questions
    python run_lego_qwen.py --max-questions 10       # smoke-test
    python run_lego_qwen.py --model-name Qwen/Qwen3-VL-30B-A3B-Instruct
"""

import argparse
import csv
import json
import os
import random
import re
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

# ── Generation constants ───────────────────────────────────────────────────
TEMPERATURE = 0.0
DO_SAMPLE = False
TOP_P = 1.0
SEED = 42


def set_seeds(seed: int = SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
from PIL import Image
from tqdm import tqdm
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# ── RunPod environment detection ───────────────────────────────────────────
# RunPod pods expose RUNPOD_POD_ID; use /workspace paths when present.
_ON_RUNPOD = bool(os.environ.get("RUNPOD_POD_ID"))
_DEFAULT_DATA_DIR = os.environ.get(
    "DATA_DIR", "/workspace/data/lego" if _ON_RUNPOD else "data/lego"
)
_DEFAULT_RESULTS_DIR = os.environ.get(
    "RESULTS_DIR", "/workspace/results" if _ON_RUNPOD else os.path.expanduser("~/results")
)
_DEFAULT_MODEL = os.environ.get("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Instruct")
_DEFAULT_MODE = os.environ.get("BENCHMARK_MODE", "lite")  # "lite" or "full"

# Log in to HuggingFace if a token is provided (needed for gated models)
_HF_TOKEN = os.environ.get("HF_TOKEN")
if _HF_TOKEN:
    from huggingface_hub import login as _hf_login
    _hf_login(token=_HF_TOKEN, add_to_git_credential=False)

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_script_dir, os.path.abspath(os.path.join(_script_dir, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from datasets.lego_dataset import (
    download_lego_tsv,
    convert_lego_to_benchmark_questions,
    LEGO_CATEGORIES,
)


# ── Expert-activation tracker ──────────────────────────────────────────────

class ExpertTracker:
    """
    Registers forward hooks on every MoE layer and accumulates the top-K
    expert indices / scores for each token during generation.

    Usage:
        tracker = ExpertTracker(model)
        tracker.begin_question(qid, category)
        model.generate(...)
        log = tracker.end_question()   # dict with all layer activations
        tracker.remove()               # clean up hooks when done
    """

    TOP_K = 8

    def __init__(self, model):
        self._hooks = []
        self._current: dict | None = None

        # Find layers regardless of where they sit in the model hierarchy
        layers = None
        for attr in ("model.layers", "model.language_model.layers",
                     "language_model.model.layers", "model.text_model.layers"):
            obj = model
            try:
                for part in attr.split("."):
                    obj = getattr(obj, part)
                layers = obj
                break
            except AttributeError:
                continue

        if layers is None:
            raise AttributeError(
                f"Cannot find transformer layers in {type(model).__name__}. "
                "Tried: model.layers, model.language_model.layers, "
                "language_model.model.layers, model.text_model.layers"
            )

        for idx, layer in enumerate(layers):
            mlp = layer.mlp
            if hasattr(mlp, "gate"):
                h = mlp.gate.register_forward_hook(self._make_hook(idx))
                self._hooks.append(h)
            elif hasattr(mlp, "experts"):
                h = mlp.register_forward_hook(self._make_hook(idx))
                self._hooks.append(h)

        n = len(self._hooks)
        print(f"Registered expert hooks on {n} MoE layer{'s' if n != 1 else ''}.")

    def _make_hook(self, layer_idx: int):
        def hook(module, input, output):
            if self._current is None:
                return
            # Gate output is a raw tensor of shape [tokens, num_experts]
            if isinstance(output, torch.Tensor):
                router_logits = output
            else:
                router_logits = getattr(output, "router_logits", None)
                if router_logits is None and isinstance(output, (tuple, list)):
                    router_logits = output[0]
            if router_logits is None or not isinstance(router_logits, torch.Tensor):
                return
            k = min(self.TOP_K, router_logits.shape[-1])
            topk_vals, topk_indices = torch.topk(router_logits, k=k, dim=-1)
            self._current["layers"].append({
                "layer": layer_idx,
                "topk_experts": topk_indices.cpu().tolist(),
                "topk_scores": topk_vals.cpu().tolist(),
            })
        return hook

    def begin_question(self, question_id: str, category: str):
        self._current = {"question_id": question_id, "category": category, "layers": []}

    def end_question(self) -> dict:
        log = self._current
        self._current = None
        return log

    def remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


# ── Answer extraction ──────────────────────────────────────────────────────

_LETTER_RE = re.compile(r"\b([A-D])\b")


def extract_answer_letter(text: str) -> str:
    """Return the first A/B/C/D found in model output, or full stripped text if none."""
    m = _LETTER_RE.search(text.upper())
    return m.group(1) if m else text.strip()


# ── Output writers ─────────────────────────────────────────────────────────

def _cat_stats(results: list) -> tuple[dict, dict]:
    cat_correct: dict[str, int] = {}
    cat_total: dict[str, int] = {}
    for r in results:
        cat = r["category"]
        cat_total[cat] = cat_total.get(cat, 0) + 1
        if r["correct"]:
            cat_correct[cat] = cat_correct.get(cat, 0) + 1
    return cat_correct, cat_total


def write_results_txt(results_dir: str, results: list, model_name: str):
    cat_correct, cat_total = _cat_stats(results)
    total_correct = sum(cat_correct.values())
    total = len(results)

    lines = [
        "=" * 60,
        f"Model : {model_name}",
        f"Total : {total} questions",
        f"Overall accuracy: {total_correct / total:.4f} ({total_correct}/{total})" if total else "Overall: N/A",
        "",
        "-------- Per-Category Accuracy --------",
    ]
    for cat in sorted(cat_total):
        c = cat_correct.get(cat, 0)
        t = cat_total[cat]
        lines.append(f"  {cat}: {c / t:.4f} ({c}/{t})")
    lines.append("=" * 60)

    path = os.path.join(results_dir, "results.txt")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


def write_execution_csv(results_dir: str, results: list):
    path = os.path.join(results_dir, "execution.csv")
    fields = ["question_id", "category", "gt", "prediction", "correct"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fields})


def write_lego_accuracy_csv(results_dir: str, results: list, model_name: str, split: str):
    """Write lego_accuracy.csv in the same format as the main pipeline."""
    cat_correct, cat_total = _cat_stats(results)
    total_correct = sum(cat_correct.values())
    total = len(results)

    row = {
        "Model": model_name,
        "split": split,
        "Overall": f"{total_correct / total:.4f}" if total else "N/A",
    }
    for cat in LEGO_CATEGORIES:
        if cat in cat_total:
            c = cat_correct.get(cat, 0)
            t = cat_total[cat]
            row[cat] = f"{c / t:.4f}"
        else:
            row[cat] = "N/A"

    path = os.path.join(results_dir, "lego_accuracy.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(row)


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Run Qwen3-VL-MoE on the LEGO benchmark with expert activation logging"
    )
    parser.add_argument(
        "--tsv-path", default=None,
        help="Path to LEGO.tsv (downloads if not provided)",
    )
    parser.add_argument(
        "--data-dir", default=_DEFAULT_DATA_DIR,
        help=f"Directory for LEGO data (default: {_DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--results-pth", default=_DEFAULT_RESULTS_DIR,
        help=f"Results output directory (default: {_DEFAULT_RESULTS_DIR})",
    )
    parser.add_argument(
        "--max-questions", default=-1, type=int,
        help="Max questions to run (-1 for all)",
    )
    parser.add_argument(
        "--lite", action="store_true",
        help="LEGO-Lite: 100 Qs each from Height, Position, Rotation, Ordering (400 total, default)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Full LEGO benchmark (all 1100 questions across all 11 categories)",
    )
    parser.add_argument(
        "--model-name", default=_DEFAULT_MODEL,
        help=f"HuggingFace model ID (default: {_DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"],
        help="Model weight dtype (default: bfloat16)",
    )
    args = parser.parse_args()

    set_seeds(SEED)

    # BENCHMARK_MODE env var acts as a default; CLI flags take precedence
    use_lite = not args.full
    if not args.full and not args.lite:
        use_lite = (_DEFAULT_MODE != "full")
    split_label = "lite" if use_lite else "full"

    # ── Step 0: Prepare data ───────────────────────────────────────────────
    print("=" * 60)
    print("Step 0: Preparing LEGO benchmark data")
    print("=" * 60)

    if args.tsv_path is None:
        args.tsv_path = os.path.join(args.data_dir, "LEGO.tsv")
        download_lego_tsv(args.tsv_path)

    annotations_path = convert_lego_to_benchmark_questions(
        args.tsv_path,
        args.data_dir,
        max_questions=args.max_questions,
        lite=use_lite,
    )
    images_dir = os.path.join(args.data_dir, "images")

    with open(annotations_path) as f:
        questions = json.load(f)["questions"]

    print(f"Loaded {len(questions)} questions ({split_label} mode)")
    if not questions:
        print("No questions found after filtering. Exiting.")
        return

    # ── Results folder ─────────────────────────────────────────────────────
    results_dir = os.path.join(
        args.results_pth,
        f"lego_qwen_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
    )
    os.makedirs(results_dir, exist_ok=True)
    expert_log_path = os.path.join(results_dir, "expert_logs.jsonl")

    # ── Step 1: Load model ─────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 1: Loading model")
    print("=" * 60)

    dtype_map = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }
    dtype = dtype_map[args.dtype]

    print(f"Model : {args.model_name}")
    print(f"Dtype : {args.dtype}")
    print(f"Device: {'cuda (' + torch.cuda.get_device_name(0) + ')' if torch.cuda.is_available() else 'cpu'}")

    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        args.model_name,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_name, trust_remote_code=True)

    # ── Step 2: Register expert hooks ──────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 2: Registering MoE expert hooks")
    print("=" * 60)
    tracker = ExpertTracker(model)

    # ── Step 3: Inference loop ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 3: Running inference with expert logging")
    print("=" * 60)

    results = []
    with open(expert_log_path, "w") as log_file:
        for q in tqdm(questions, desc="Evaluating"):
            qid = f"{q['image_index']}_{q['question_index']}"
            image_path = os.path.join(images_dir, q["image_filename"])
            image = Image.open(image_path).convert("RGB")

            tracker.begin_question(qid, q["category"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": q["question"]},
                    ],
                }
            ]
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text_prompt],
                images=[image],
                return_tensors="pt",
                padding=True,
            ).to(model.device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=32,
                    do_sample=DO_SAMPLE,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                )

            expert_log = tracker.end_question()

            # Decode only the newly generated tokens (strip the input prompt)
            input_len = inputs["input_ids"].shape[1]
            new_tokens = generated_ids[0][input_len:]
            raw_text = processor.decode(new_tokens, skip_special_tokens=True)

            prediction = extract_answer_letter(raw_text)
            gt = q["answer"].strip().upper()
            correct = prediction.upper() == gt

            result = {
                "question_id": qid,
                "category": q["category"],
                "gt": gt,
                "prediction": prediction,
                "correct": correct,
            }
            results.append(result)

            # Write incrementally so a crash doesn't lose completed work
            log_file.write(json.dumps({**result, "expert_log": expert_log}) + "\n")

    tracker.remove()

    # ── Step 4: Write summary files ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Step 4: Writing results")
    print("=" * 60)

    write_execution_csv(results_dir, results)
    write_lego_accuracy_csv(results_dir, results, args.model_name, split_label)
    write_results_txt(results_dir, results, args.model_name)

    print(f"\nResults saved to : {results_dir}")
    print(f"Expert logs      : {expert_log_path}")


if __name__ == "__main__":
    main()
