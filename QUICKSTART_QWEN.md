# Qwen Quickstart

This project now has two different "model jobs":

1. The text model writes signatures, helper functions, and programs.
2. The vision backend answers questions about the LEGO image itself.

For your Parasail setup, the text model is the important part. It now defaults to `Qwen2.5-72B-Instruct`.

## What you need

Before you run anything, make sure you have:

- Python installed
- A Parasail API key
- A GPU if you want the full vision pipeline to be fast

## Step 1: Open the project folder

From Terminal:

```bash
cd /Users/erosmendoza/Downloads/BergLabResearch-master/lego-spatial-pipeline
```

## Step 2: Create a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate
```

## Step 3: Install dependencies

```bash
pip install -r requirements.txt
pip install pandas
```

## Step 4: Tell the project to use Parasail + Qwen

Set these environment variables in the same terminal window:

```bash
export PARASAIL_API_KEY="your_parasail_key_here"
export PARASAIL_BASE_URL="https://api.parasail.io/v1"
export PARASAIL_MODEL_NAME="Qwen2.5-72B-Instruct"
```

Optional:

If Parasail gives you a deployment name or alias instead of a raw model name, use this instead:

```bash
export PARASAIL_DEPLOYMENT_NAME="your_exact_parasail_deployment_name"
```

Optional:

```bash
export VISION_BACKEND_MODEL="gpt-4o"
```

Use `VISION_BACKEND_MODEL` only for the image-understanding parts of the pipeline.

## Step 5: Run a tiny test first

This is the safest first run:

```bash
python3 run_lego.py --max-questions 5 --stub
```

What this does:

- Uses LEGO-Lite by default
- Converts the dataset into the local question format
- Runs only a tiny sample
- Skips the heavy local vision models because of `--stub`

## Step 6: Run LEGO-Lite

When the small test looks good:

```bash
python3 run_lego.py
```

That now means:

- LEGO-Lite only
- text-only MCQ / true-false filtering
- Qwen via Parasail for the text-generation steps

## Step 7: Run the full benchmark

```bash
python3 run_lego.py --full
```

## What files you should look at after a run

- `data/lego/annotations.json`: the converted LEGO questions
- `results/.../signature_generator/`: signature-generation traces
- `results/.../api_generator/`: helper-method traces
- `results/.../program_generator/`: generated programs
- `results/.../program_execution/execution.json`: final predictions
- `results/.../program_execution/lego_accuracy.csv`: category accuracy summary

## Very simple mental model

Think of the pipeline like this:

1. It reads LEGO questions.
2. Qwen writes a plan in code for how to answer them.
3. The runtime executes that code against the LEGO image.
4. The final output is usually one letter like `A`, `B`, `C`, or `D`.

## Important note

Qwen via Parasail is used for the text-generation side.
The actual image-question-answering path is still controlled separately by `VISION_BACKEND_MODEL`.
