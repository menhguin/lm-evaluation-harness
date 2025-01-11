import json
from typing import Union, Dict, List, Optional, Any
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
from lm_eval.api.instance import Instance
from lm_eval.models.utils import handle_stop_sequences
from tqdm import tqdm


eval_logger = utils.eval_logger


class GoodfireLLM(LM):
    """
    Integration of Goodfire's 'OpenAI-plus' chat API with the EleutherAI lm-eval-harness.
    This class uses the Goodfire client for completions.

    Limitations:
      - We do not currently support loglikelihood or loglikelihood_rolling.
      - Only 'generate_until' style completions are provided (i.e., chat-based completions).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "meta-llama/Meta-Llama-3-8B-Instruct",
        max_completion_tokens: int = 512,
        temperature: float = 1.0,
    ):
        """
        Args:
          api_key: str. Your Goodfire API key. If not provided, tries environment or user-provided secrets.
          model: str. The Goodfire model or variant to use, e.g. "meta-llama/Meta-Llama-3-8B-Instruct".
          max_completion_tokens: int. Max tokens to generate when calling the Goodfire API.
          temperature: float for sampling temperature.
        """
        super().__init__()
        self.api_key = api_key or os.getenv("GOODFIRE_API_KEY")
        if not self.api_key:
            raise ValueError("Goodfire API key is required but not found")
        
        self.client = goodfire.Client(api_key=self.api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._max_length = 4096  # Default max length for context + completion

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self.max_completion_tokens

    def tok_encode(self, string: str) -> List[int]:
        # Placeholder for tokenization - actual token count not needed
        return [0] * (len(string) // 4)  # Rough estimate

    def tok_decode(self, tokens: List[int]) -> str:
        # Not needed for generation
        return ""

    def _model_call(self, inps):
        """
        Not used by lm-eval-harness for generate_until; raise if accidentally called.
        """
        raise NotImplementedError("GoodfireLLM does not support direct _model_call usage.")

    def _model_generate(self, context, max_length, eos_token_id):
        """
        Not used, we override 'generate_until' from LocalCompletionsAPI.
        """
        raise NotImplementedError("GoodfireLLM does not use _model_generate in this integration.")

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not supported for Goodfire models")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Loglikelihood_rolling not supported for Goodfire models")

    def format_results(self, results):
        """Format results in a clean table format similar to VLLM output."""
        output = []
        
        # Add model info
        output.append(f"\nResults for Goodfire LLM Evaluation:")
        output.append("-" * 100)
        output.append(f"Model: goodfire (model={self.model})")
        output.append(f"Temperature: {self.temperature}")
        output.append("-" * 100)
        
        # Header
        output.append("|     Tasks     |Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|")
        output.append("|---------------|------:|----------------|-----:|-----------|---|-----:|---|-----:|")
        
        # Results for each task
        if 'results' in results:
            for task_name, task_results in results['results'].items():
                version = results['versions'].get(task_name, '')
                n_shot = results['n-shot'].get(task_name, 0)
                
                # Handle different metrics and filters
                for key in task_results:
                    if ',' in str(key) and key != 'alias':
                        metric, filter_name = key.split(',', 1)
                        if not metric.endswith('_stderr'):
                            value = task_results[key]
                            stderr = task_results.get(f'{metric}_stderr,{filter_name}', 0)
                            
                            # Get higher_is_better info
                            higher_is_better = results.get('higher_is_better', {}).get(task_name, {}).get(metric)
                            arrow = "↑" if higher_is_better else "↓" if higher_is_better is False else " "
                            
                            output.append(
                                f"|{task_name:<15}|{version:>6}|{filter_name:<14}|{n_shot:>5}|{metric:>10}|{arrow}  |{value:>5.4f}|±  |{stderr:>5.4f}|"
                            )
        
        output.append("-" * 100)
        formatted = "\n".join(output)
        eval_logger.info(formatted)
        return formatted

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        res = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="Running requests")

        for req in requests:
            context, gen_kwargs = req.args
            if isinstance(gen_kwargs, dict):
                kwargs = gen_kwargs.copy()
                until = handle_stop_sequences(kwargs.pop("until", []), eos=None)
                top_p = kwargs.pop("top_p", 1.0)
                temperature = kwargs.pop("temperature", self.temperature)
            else:
                until = []
                top_p = 1.0
                temperature = self.temperature

            # Log the first prompt for debugging
            if len(res) == 0:
                print(f"\nFirst prompt: {context}\n")

            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": context}],
                    model=self.model,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=temperature,
                    top_p=top_p,
                )
                
                # Log the first response for debugging
                if len(res) == 0:
                    print(f"\nFirst response: {response.choices[0].message.content}\n")

                output = response.choices[0].message.content
                
                # Handle stop sequences if provided
                if until:
                    for stop_seq in until:
                        if stop_seq in output:
                            output = output[:output.index(stop_seq)]

                res.append(output)
                pbar.update(1)
            except Exception as e:
                print(f"Error generating response: {str(e)}")
                res.append("")  # Return empty string on error
                pbar.update(1)

        pbar.close()
        return res

# Register the model
def create_model(pretrained='meta-llama/Meta-Llama-3-8B-Instruct', **kwargs):
    return GoodfireLLM(model=pretrained, **kwargs)

if __name__ == "__main__":
    from lm_eval import tasks, evaluator
    from lm_eval.api.registry import register_model

    register_model("goodfire_llms", create_model)