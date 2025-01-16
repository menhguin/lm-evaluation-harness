import os
import argparse
from dotenv import load_dotenv
import huggingface_hub
import goodfire
import wandb

def setup_api_keys():
    """Load and setup API keys from .env file"""
    load_dotenv()
    
    # Setup Goodfire
    goodfire_key = os.getenv('GOODFIRE_API_KEY')
    if not goodfire_key:
        raise ValueError("GOODFIRE_API_KEY not found in .env")
    os.environ['GOODFIRE_API_KEY'] = goodfire_key
    
    # Setup HuggingFace
    hf_token = os.getenv('HF_READ_TOKEN')
    if hf_token:
        huggingface_hub.login(hf_token)
    else:
        print("HF_READ_TOKEN not found. Some tasks like GPQA may not work.")
    
    # Setup WandB (optional)
    wandb_token = os.getenv('WANDB_API_KEY')
    if wandb_token:
        os.environ["WANDB_API_KEY"] = wandb_token
        wandb.login()
    else:
        print("WANDB_API_KEY not found. Continuing without WandB logging.")

def test_goodfire_connection():
    """Test Goodfire API connection"""
    client = goodfire.Client()
    response = client.chat.completions.create(
        messages=[{"role": "user", "content": "Say hello!"}],
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        max_completion_tokens=10
    )
    print("Test response:", response.choices[0].message['content'])

def run_evaluation(args):
    """Run the evaluation with specified parameters"""
    cmd = f"""lm_eval \
        --model {args.model} \
        --model_args pretrained={args.model_args} \
        --batch_size "auto" \
        --tasks {args.tasks} \
        --num_fewshot {args.num_fewshot} \
        --apply_chat_template \
        --fewshot_as_multiturn \
        --limit {args.limit} \
        --log_samples \
        --output_path {args.output_path} \
        --gen_kwargs {args.sampler}={args.sampler_value},temperature={args.temperature},do_sample=True \
        --device {args.device}"""
    
    if args.wandb_logging:
        wandb_name = f"{args.tasks}_{args.sampler}_{args.sampler_value}_temp_{args.temperature}_{args.model}_{args.model_args.replace('/', '_')}"
        cmd += f' --wandb_args project=lm-eval-harness-integration,name={wandb_name}'
    
    os.system(cmd)

def main():
    parser = argparse.ArgumentParser(description='Run LM Evaluation')
    parser.add_argument('--model', default='goodfire', help='Model type (goodfire/vllm)')
    parser.add_argument('--model_args', default='meta-llama/Meta-Llama-3.1-8B-Instruct', help='Model path/name')
    parser.add_argument('--tasks', default='gsm8k_cot_llama', help='Task to evaluate')
    parser.add_argument('--num_fewshot', default='8', help='Number of few-shot examples')
    parser.add_argument('--sampler', default='top_p', help='Sampling method')
    parser.add_argument('--sampler_value', default='0.9', help='Sampling parameter value')
    parser.add_argument('--temperature', default=1, type=float, help='Temperature')
    parser.add_argument('--limit', default=30, type=int, help='Limit number of examples')
    parser.add_argument('--output_path', default='./lm-eval-output/', help='Output path')
    parser.add_argument('--device', default='cuda', help='Device to use')
    parser.add_argument('--wandb_logging', action='store_true', help='Enable WandB logging')
    parser.add_argument('--test_connection', action='store_true', help='Test Goodfire API connection')
    
    args = parser.parse_args()
    
    # Setup environment
    setup_api_keys()
    
    # Test connection if requested
    if args.test_connection:
        test_goodfire_connection()
        return
    
    # Run evaluation
    run_evaluation(args)

if __name__ == "__main__":
    main() 