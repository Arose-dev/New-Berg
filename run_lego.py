"""
Run the program-synthesis pipeline on the LEGO-Puzzles benchmark.

This script runs the 4-step text-to-program pipeline
(signature → API → program → execution) and outputs per-category accuracy.

For direct Qwen3-VL-MoE inference with expert activation logging, use
run_lego_qwen.py instead — it shares the same --lite / --full flags.

Usage:
    # LEGO-Lite (default): 100 Qs each from Height, Position, Rotation, Ordering = 400 total
    python run_lego.py

    # Full benchmark (all 1100 questions across all 11 categories)
    python run_lego.py --full

    # Small test (20 questions)
    python run_lego.py --max-questions 20

    # Use pre-downloaded TSV
    python run_lego.py --tsv-path data/lego/LEGO.tsv

    # Skip heavy model loading (for testing pipeline)
    python run_lego.py --max-questions 5 --stub
"""

import argparse
import os
import sys

_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_script_dir, os.path.abspath(os.path.join(_script_dir, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
import torch
import numpy as np
from datetime import datetime
import json

from agents.agents import SignatureAgent, APIAgent, ProgramAgent
from engine.engine import Engine
from prompts.modules import MODULES_SIGNATURES_LEGO
from datasets.lego_dataset import (
    download_lego_tsv,
    convert_lego_to_benchmark_questions,
)


def check_cuda():
    """Verify CUDA is available before running the pipeline."""
    if not torch.cuda.is_available():
        print("=" * 60)
        print("WARNING: CUDA is NOT available!")
        print("This pipeline runs best on an NVIDIA GPU with CUDA.")
        print("Running on CPU will be extremely slow.")
        print("=" * 60)
        response = input("Continue anyway? (y/n): ")
        if response.lower() != "y":
            sys.exit(1)
    else:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"CUDA OK: {gpu_name} ({gpu_mem:.1f} GB VRAM)")


def set_seeds(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)


def main():
    set_seeds(42)

    parser = argparse.ArgumentParser(description="Run the LEGO spatial reasoning pipeline")
    parser.add_argument(
        "--tsv-path", default=None,
        help="Path to LEGO.tsv (downloads if not provided)",
    )
    parser.add_argument(
        "--data-dir", default="data/lego",
        help="Directory for LEGO data (default: data/lego)",
    )
    parser.add_argument(
        "--models-path", default="models/",
        help="Path to vision models directory",
    )
    parser.add_argument(
        "--results-pth", default=os.path.expanduser("~/results"),
        help="Path to save results (default: ~/results)",
    )
    parser.add_argument(
        "--max-questions", default=-1, type=int,
        help="Max questions to evaluate (-1 for all)",
    )
    parser.add_argument(
        "--num-api-questions", default=10, type=int,
        help="Number of questions for API generation",
    )
    parser.add_argument(
        "--lite", action="store_true",
        help="Use LEGO-Lite: 100 questions each from Height, Position, Rotation, Ordering (400 total, default)",
    )
    parser.add_argument(
        "--full", action="store_true",
        help="Use the full LEGO benchmark (all 1100 questions across all 11 categories)",
    )
    parser.add_argument(
        "--stub", action="store_true",
        help="Skip loading Molmo (for testing pipeline)",
    )
    args = parser.parse_args()

    # Pre-flight check
    if not args.stub:
        check_cuda()

    # Step 0: Download and convert LEGO data
    print("=" * 60)
    print("Step 0: Preparing LEGO benchmark data")
    print("=" * 60)

    if args.tsv_path is None:
        args.tsv_path = os.path.join(args.data_dir, "LEGO.tsv")
        download_lego_tsv(args.tsv_path)

    use_lite = True
    if args.full:
        use_lite = False
    elif args.lite:
        use_lite = True

    annotations_path = convert_lego_to_benchmark_questions(
        args.tsv_path,
        args.data_dir,
        max_questions=args.max_questions,
        lite=use_lite,
    )
    images_path = os.path.join(args.data_dir, "images")

    # Load questions
    with open(annotations_path, "r") as f:
        questions_data = json.load(f)
    questions = list(questions_data["questions"])
    if args.max_questions > 0:
        questions = questions[:args.max_questions]

    print(f"Loaded {len(questions)} questions")

    if not questions:
        print("No usable LEGO questions were found after filtering.")
        print("Try a larger sample size or inspect data/lego/annotations.json.")
        return

    # Create results folder
    results_folder_path = os.path.join(
        args.results_pth,
        f"lego_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(results_folder_path)

    signature_agent = SignatureAgent(MODULES_SIGNATURES_LEGO)

    if args.stub:
        print("\n" + "=" * 60)
        print("Step 1-2: Stub Mode")
        print("=" * 60)
        print("Skipping signature/API generation and using predefined LEGO tools only.")
        api_agent = APIAgent(signature_agent, "lego", api=[])
    else:
        # Step 1: Generate signatures
        print("\n" + "=" * 60)
        print("Step 1: Generating API Signatures")
        print("=" * 60)

        api_questions = random.sample(
            questions, min(args.num_api_questions, len(questions))
        )
        signature_agent.get_signatures(
            api_questions, images_path, results_folder_path,
        )

        # Step 2: Implement API methods
        print("\n" + "=" * 60)
        print("Step 2: Generating API Implementations")
        print("=" * 60)

        api_agent = APIAgent(signature_agent, "lego")
        api_agent.get_api_implementations(results_folder_path)

    # Step 3: Generate solution programs
    print("\n" + "=" * 60)
    print("Step 3: Generating Programs")
    print("=" * 60)

    program_agent = ProgramAgent(api_agent, dataset="lego")
    program_agent.get_programs(
        questions, images_path, results_folder_path,
    )

    # Step 4: Execute programs
    print("\n" + "=" * 60)
    print("Step 4: Executing Programs")
    print("=" * 60)

    engine = Engine(
        api_agent.api,
        results_folder_path=results_folder_path,
        models_path=args.models_path,
        dataset="lego",
        stub=args.stub,
    )
    engine.execute_programs(
        program_agent.programs,
        questions,
        images_path,
    )

    print("\n" + "=" * 60)
    print("DONE!")
    print(f"Results saved to: {results_folder_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
