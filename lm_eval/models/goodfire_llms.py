import json
from typing import Union, Dict, List, Optional
import asyncio
import os

try:
    import goodfire
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Please install goodfire to use the GoodfireLLM: pip install goodfire"
    ) from e

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.models.openai_completions import LocalCompletionsAPI


eval_logger = utils.eval_logger


class GoodfireLLM(LM):
    """
    Integration of Goodfire's 'OpenAI-plus' chat API with the EleutherAI lm-eval-harness.
    This class uses the Goodfire client for completions.

    Limitations:
      - We do not currently support loglikelihood or loglikelihood_rolling.
      - Only 'generate_until' style completions are provided (i.e., chat-based completions).
    """

    def __init__(self, api_key=None, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_completion_tokens=50, temperature=0.7):
        super().__init__()
        self.api_key = api_key or os.getenv("GOODFIRE_API_KEY")
        if not self.api_key:
            raise ValueError("Goodfire API key not found. Please provide it as an argument or set GOODFIRE_API_KEY environment variable.")
        self.client = goodfire.Client(api_key=self.api_key)
        self.model_name = model_name
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._max_length = None

    def format_results(self, results):
        """Format results in a clean table format."""
        formatted_output = []
        formatted_output.append("\nResults for Goodfire LLM Evaluation:")
        formatted_output.append("-" * 100)
        formatted_output.append(f"{'Task':<30} {'Version':<10} {'Filter':<15} {'N-shot':<8} {'Metric':<15} {'Value':<10} {'Stderr':<10}")
        formatted_output.append("-" * 100)
        
        for task_name, task_results in results.items():
            if task_name == "config":
                continue
                
            # Get task metadata
            version = task_results.get("version", "N/A")
            n_shot = task_results.get("num_fewshot", 0)
            
            # Process metrics
            metrics = {}
            for key, value in task_results.items():
                if "," in str(key) and key != "alias":
                    metric, filter_name = key.split(",", 1)
                    if not metric.endswith("_stderr"):
                        metrics[key] = {
                            "metric": metric,
                            "filter": filter_name,
                            "value": value,
                            "stderr": task_results.get(f"{metric}_stderr,{filter_name}", "N/A")
                        }
            
            # Add rows for each metric
            for key, data in metrics.items():
                formatted_output.append(
                    f"{task_name:<30} {version:<10} {data['filter']:<15} {n_shot:<8} "
                    f"{data['metric']:<15} {data['value']:<10.4f} {data['stderr']:<10}"
                )
        
        formatted_output.append("-" * 100)
        return "\n".join(formatted_output)

    def generate_until(self, requests):
        if not requests:
            return []

        results = []
        for request in requests:
            prompt = request["prompt"]
            until = request.get("until", [])

            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=self.model_name,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=self.temperature
                )
                
                # Extract the generated text from the response
                generated_text = response.choices[0].message.content
                
                # Find the first occurrence of any until sequence
                min_idx = len(generated_text)
                first_until = None
                
                for u in until:
                    idx = generated_text.find(u)
                    if idx != -1 and idx < min_idx:
                        min_idx = idx
                        first_until = u
                
                if first_until is not None:
                    generated_text = generated_text[:min_idx] + first_until
                
                results.append(generated_text)
                
            except Exception as e:
                print(f"Error in generate_until: {str(e)}")
                results.append("")  # Return empty string on error
        
        # If this is part of an evaluation (indicated by the presence of task-specific attributes)
        if hasattr(self, '_current_task_list') and hasattr(self, '_current_results'):
            formatted_output = []
            formatted_output.append("\nResults for Goodfire LLM Evaluation:")
            formatted_output.append("-" * 100)
            formatted_output.append(f"{'Task':<30} {'Version':<10} {'Filter':<15} {'N-shot':<8} {'Metric':<15} {'Value':<10} {'Stderr':<10}")
            formatted_output.append("-" * 100)
            
            for task_name, task_results in self._current_results.items():
                if task_name == "config":
                    continue
                    
                # Get task metadata
                version = task_results.get("version", "N/A")
                n_shot = task_results.get("num_fewshot", 0)
                
                # Process metrics
                metrics = {}
                for key, value in task_results.items():
                    if "," in str(key) and key != "alias":
                        metric, filter_name = key.split(",", 1)
                        if not metric.endswith("_stderr"):
                            metrics[key] = {
                                "metric": metric,
                                "filter": filter_name,
                                "value": value,
                                "stderr": task_results.get(f"{metric}_stderr,{filter_name}", "N/A")
                            }
                
                # Add rows for each metric
                for key, data in metrics.items():
                    formatted_output.append(
                        f"{task_name:<30} {version:<10} {data['filter']:<15} {n_shot:<8} "
                        f"{data['metric']:<15} {data['value']:<10.4f} {data['stderr']:<10}"
                    )
            
            formatted_output.append("-" * 100)
            print("\n".join(formatted_output))
        
        return results

    @property
    def max_length(self):
        if self._max_length is None:
            # Set a default max length if not specified
            self._max_length = 2048
        return self._max_length

    @max_length.setter
    def max_length(self, value):
        self._max_length = value

    def tokenize(self, requests):
        # For simplicity, just split into words
        # In a real implementation, you'd want to use a proper tokenizer
        return [request.split() for request in requests]

    def encode(self, request):
        """Encode the request into tokens."""
        # For simplicity, just split into words
        # In a real implementation, you'd want to use a proper tokenizer
        return request.split()

    def decode(self, tokens):
        """Decode tokens back into text."""
        # For simplicity, just join words
        # In a real implementation, you'd want to use a proper detokenizer
        return " ".join(tokens)

    def loglikelihood(self, requests):
        """Not implemented for this model."""
        raise NotImplementedError("Loglikelihood calculation is not supported for this model.")

    def loglikelihood_rolling(self, requests):
        """Not implemented for this model."""
        raise NotImplementedError("Rolling loglikelihood calculation is not supported for this model.")