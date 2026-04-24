# LEGO-Puzzles Spatial Reasoning Pipeline

A program synthesis pipeline for evaluating spatial reasoning on the [LEGO-Puzzles benchmark](https://huggingface.co/datasets/). Given an image of a LEGO scene and a question, the pipeline generates and executes a Python program to produce an answer.

## How It Works

The pipeline runs in four stages:

1. **Signature Agent** — generates function signatures for tools needed to answer the question
2. **API Agent** — implements those tool functions using vision models
3. **Program Agent** — writes a program using the generated API to answer the question
4. **Execution** — runs the program and scores the result against the ground truth

The LLM (configurable via environment variables) handles all text/code generation. Vision tasks use SAM2, UniDepth, and GroundingDINO loaded locally.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
sh setup.sh
```

Requires CUDA 12.2 and Python 3.10. For other CUDA versions, update the `--index-url` in `setup.sh` accordingly.

## Configuration

Set the following environment variables before running (no config file needed):

| Variable | Description | Example |
|---|---|---|
| `API_KEY` | Your LLM provider API key | `export API_KEY=...` |
| `API_BASE_URL` | Your provider's OpenAI-compatible base URL | `export API_BASE_URL=https://api.fireworks.ai/inference/v1` |
| `MODEL_NAME` | Model to use for text/code generation | `export MODEL_NAME=accounts/fireworks/models/llama-v3p1-70b-instruct` |

Any OpenAI-compatible API works (Parasail, Fireworks, OpenAI, etc.).

## Running

```bash
# LEGO-Lite subset (default)
python run_lego.py

# Full benchmark (~1100 questions)
python run_lego.py --full

# Small test run
python run_lego.py --max-questions 20

# Use a pre-downloaded TSV
python run_lego.py --tsv-path data/lego/LEGO.tsv

# Skip vision model loading (for testing the pipeline only)
python run_lego.py --max-questions 5 --stub
```

## Output

Results are saved to `~/results/lego_<timestamp>/`:

```
lego_<timestamp>/
├── program_generator/       # per-question generated programs (.html)
├── program_execution/       # per-question execution traces and outputs
├── execution.csv            # full execution log
└── results.txt              # per-category accuracy summary
```
