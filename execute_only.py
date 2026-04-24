"""
Script to run only the 'Executing programs' step using saved results
from a previous run, skipping the API-heavy signature/API/program generation.
"""

# argparse: lets us accept command-line arguments like --dataset clevr
import argparse
# os: for file path operations (joining paths, checking if files exist, etc.)
import os
# sys: for modifying Python's module search path so we can import from parent dirs
import sys
# random: Python's built-in random number generator (we lock it for reproducibility)
import random
# torch: PyTorch deep learning library (used by SAM2, UniDepth, GroundingDINO)
import torch
# numpy: math/array library (also needs its random seed locked)
import numpy as np
# json: for reading and writing JSON files (api.json, programs.json, annotations.json)
import json

# Project root (script dir) and parent in path so imports work from any cwd
_script_dir = os.path.dirname(os.path.abspath(__file__))
for _p in (_script_dir, os.path.abspath(os.path.join(_script_dir, ".."))):
    if _p not in sys.path:
        sys.path.insert(0, _p)
module_path = _script_dir

# Import the Engine class -- this is what actually runs programs against images
# using the vision models (SAM2, UniDepth, GroundingDINO, Molmo)
from engine.engine import Engine


def set_seeds(seed):
    """Lock all random number generators to the same seed for reproducible results."""
    # Lock Python's built-in random number generator
    random.seed(seed)
    # Lock PyTorch's random number generator (used by the vision models)
    torch.manual_seed(seed)
    # Lock NumPy's random number generator (used by array operations)
    np.random.seed(seed)


def run_execution(args):
    """Main function: loads saved results and runs only the execution step."""

    # Open the annotations file (contains all the CLEVR questions + ground truth answers)
    with open(args.annotations_json, "r") as file:
        # Parse the JSON file into a Python dictionary
        questions_data = json.load(file)
    # Get the list of questions; [:args.num_questions] limits how many we use (-1 = all)
    questions = list(questions_data["questions"])[: args.num_questions]

    # Open the saved api.json from a PREVIOUS run
    # This contains the custom methods that GPT-4o generated (e.g. _find_object_by_color)
    # Loading from disk means we DON'T need to call OpenAI again
    with open(args.api_json, "r") as file:
        api = json.load(file)

    # Open the saved programs.json from a PREVIOUS run
    # This contains the 1,154 solution programs GPT-4o wrote (one per question)
    # Loading from disk means we DON'T need to call OpenAI again
    with open(args.programs_json, "r") as file:
        programs = json.load(file)

    # Print a summary so we can verify everything loaded correctly
    print(f"Loaded {len(api)} API methods and {len(programs)} programs from previous run.")
    print(f"Questions: {len(questions)}")

    # Create the Engine -- this is the heavy step
    # It loads SAM2, UniDepth, GroundingDINO, and Molmo into memory
    # Use --stub to skip loading Molmo (avoids crash on some systems); answers will be placeholder.
    print("Executing programs...")
    engine = Engine(
        api,                              # the saved API methods from the previous run
        results_folder_path=args.results_pth,  # where to save execution results
        models_path=args.models_path,     # where the vision models are stored (models/)
        dataset=args.dataset,             # which dataset we're using (clevr)
        stub=getattr(args, "stub", False),  # skip Molmo to avoid load crash
    )

    # Run all 1,154 programs against the actual CLEVR images
    # For each question, it takes the program GPT-4o wrote, executes it,
    # and the program calls vision model functions to answer the question
    engine.execute_programs(
        programs,                         # the 1,154 saved programs
        questions,                        # the 1,154 questions (with ground truth answers)
        args.image_pth,                   # path to the CLEVR images folder
        oracle=args.oracle,               # if True, use ground truth instead of models
        scenes_json_path=args.scenes_json,  # ground truth scene data (only with oracle)
    )


# This block only runs when you execute this file directly (not when imported)
if __name__ == "__main__":
    # Set all random seeds to 42 for reproducibility
    set_seeds(42)

    # Create an argument parser to handle command-line options
    parser = argparse.ArgumentParser()

    # --dataset: which dataset to evaluate on (defaults to clevr)
    parser.add_argument(
        "--dataset", default="clevr", choices=["gqa", "clevr", "omni3d", "lego"],
    )

    # --annotations-json: path to the questions file (defaults to clevr_subset)
    parser.add_argument(
        "--annotations-json",
        default="data/clevr_subset/annotations.json",
    )

    # --image-pth: path to the folder containing the CLEVR images
    parser.add_argument(
        "--image-pth",
        default="data/clevr_subset/images/",
    )

    # --models-path: where SAM2, UniDepth, GroundingDINO are stored
    parser.add_argument(
        "--models-path",
        default="models/",
    )

    # --results-pth: (REQUIRED) path to the previous run's results folder
    # This is where execution results will also be saved
    parser.add_argument(
        "--results-pth",
        required=True,
        help="Path to existing results folder (e.g. results/2026-02-07_20-49-20)",
    )

    # --api-json: (REQUIRED) path to the saved api.json from a previous run
    # Contains the GPT-4o-generated methods so we don't need to regenerate them
    parser.add_argument(
        "--api-json",
        required=True,
        help="Path to saved api.json (e.g. results/2026-02-07_20-49-20/api_generator/api.json)",
    )

    # --programs-json: (REQUIRED) path to the saved programs.json from a previous run
    # Contains the GPT-4o-generated solution programs so we don't need to regenerate them
    parser.add_argument(
        "--programs-json",
        required=True,
        help="Path to saved programs.json (e.g. results/2026-02-07_20-49-20/program_generator/programs.json)",
    )

    # --scenes-json: path to ground truth scene data (only needed if using --oracle)
    parser.add_argument("--scenes-json", default="")

    # --oracle: if this flag is passed, use ground truth data instead of vision models
    parser.add_argument("--oracle", action="store_true")

    # --stub: skip loading Molmo (avoids crash on some Windows/GPU setups); loc() returns center point, rest runs
    parser.add_argument("--stub", action="store_true", help="Skip Molmo load; execution completes with placeholder locate.")

    # --num-questions: how many questions to run (-1 means all 1,154)
    parser.add_argument("--num-questions", default=-1, type=int)

    # Parse all the command-line arguments into an object we can use
    args = parser.parse_args()

    # Run the execution step with the parsed arguments
    run_execution(args)
