# Local LM Evaluation Harness Setup

This repository contains a local setup for running the LM Evaluation Harness with Goodfire and other models.

## Setup

1. Create and activate a virtual environment:
```bash
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Unix/MacOS:
source venv/bin/activate
```

2. Install dependencies:
```bash
pip install -e .
pip install goodfire vllm wandb huggingface_hub python-dotenv
```

3. Create a `.env` file in the root directory with your API keys:
```
GOODFIRE_API_KEY=your_goodfire_api_key_here
HF_READ_TOKEN=your_huggingface_token_here
WANDB_API_KEY=your_wandb_api_key_here  # Optional for logging
```

## Usage

The main script `run_eval.py` provides a command-line interface for running evaluations. Here are some example commands:

1. Test Goodfire API connection:
```bash
python run_eval.py --test_connection
```

2. Run GSM8K evaluation with default settings:
```bash
python run_eval.py --tasks gsm8k_cot_llama
```

3. Run GPQA evaluation with custom settings:
```bash
python run_eval.py \
    --tasks gpqa_main_generative_n_shot \
    --num_fewshot 5 \
    --wandb_logging
```

### Available Arguments

- `--model`: Model type (goodfire/vllm) [default: goodfire]
- `--model_args`: Model path/name [default: meta-llama/Meta-Llama-3.1-8B-Instruct]
- `--tasks`: Task to evaluate [default: gsm8k_cot_llama]
- `--num_fewshot`: Number of few-shot examples [default: 8]
- `--sampler`: Sampling method [default: top_p]
- `--sampler_value`: Sampling parameter value [default: 0.9]
- `--temperature`: Temperature [default: 1]
- `--limit`: Limit number of examples [default: 30]
- `--output_path`: Output path [default: ./lm-eval-output/]
- `--device`: Device to use [default: cuda]
- `--wandb_logging`: Enable WandB logging [flag]
- `--test_connection`: Test Goodfire API connection [flag]
