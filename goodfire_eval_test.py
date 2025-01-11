# Cell 1: Setup and Installation
!pip install goodfire
!pip install -e git+https://github.com/EleutherAI/lm-evaluation-harness.git#egg=lm_eval[vllm]

import os
from google.colab import userdata  # Only needed if running in Colab

# Try to get API key from environment or Colab secrets
GOODFIRE_API_KEY = os.getenv('GOODFIRE_API_KEY') or userdata.get('GOODFIRE_API_KEY')
if not GOODFIRE_API_KEY:
    raise ValueError("Please set GOODFIRE_API_KEY in environment or Colab secrets")

print("Setup complete!")

# Cell 2: Test Goodfire Client
import goodfire

client = goodfire.Client(api_key=GOODFIRE_API_KEY)

# Simple test call
response = client.chat.completions.create(
    messages=[{"role": "user", "content": "Say hello!"}],
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_completion_tokens=10
)

print("Test response:", response.choices[0].message.content)

# Cell 3: Create Test Task
from lm_eval.api.task import Task
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from typing import List, Any, Dict

class SimpleGenerationTask(Task):
    def __init__(self):
        super().__init__()
        
    def get_instances(self) -> List[Instance]:
        # Create two simple test prompts
        prompts = [
            "What is 2+2?",
            "Write a haiku about programming."
        ]
        return [Instance(prompt=p) for p in prompts]
        
    def evaluate_instances(self, instances: List[Instance]) -> Dict[str, Any]:
        # Just collect the generations
        return {
            "generations": [instance.output for instance in instances],
            "prompts": [instance.prompt for instance in instances]
        }

    def aggregation(self) -> Dict[str, Any]:
        # No aggregation needed for this test
        return {}

    def higher_is_better(self) -> Dict[str, bool]:
        return {}

print("Test task created!")

# Cell 4: Run Test Evaluation
from lm_eval.models.goodfire_llms import GoodfireLLM
from lm_eval.evaluator import evaluate

# Initialize our model
model = GoodfireLLM(
    api_key=GOODFIRE_API_KEY,
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_completion_tokens=50,
    temperature=0.7
)

# Create and run task
task = SimpleGenerationTask()
results = evaluate(
    model=model,
    tasks=[task],
    num_fewshot=0,
    batch_size=1
)

print("\nEvaluation Results:")
print("==================")
for prompt, generation in zip(
    results["simple_generation"]["prompts"],
    results["simple_generation"]["generations"]
):
    print(f"\nPrompt: {prompt}")
    print(f"Generation: {generation}")

# Cell 5: Run GSM8K Task (with limit)
from lm_eval import evaluator

# Initialize model with different settings for math
model = GoodfireLLM(
    api_key=GOODFIRE_API_KEY,
    model="meta-llama/Meta-Llama-3-8B-Instruct",
    max_completion_tokens=512,  # Longer for math reasoning
    temperature=0.1  # Lower temp for math
)

# Run evaluation on just 2 examples to test
results = evaluator.simple_evaluate(
    model=model,
    tasks=["gsm8k"],
    num_fewshot=8,
    limit=2  # Just test 2 examples
)

print("\nGSM8K Results:")
print(results) 