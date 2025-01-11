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

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        max_completion_tokens: int = 128,
        temperature: float = 0.0,
        **kwargs,
    ):
        """
        Args:
          api_key: str. Your Goodfire API key. If not provided, tries environment or user-provided secrets.
          model: str. The Goodfire model or variant to use, e.g. "meta-llama/Meta-Llama-3-8B-Instruct".
          max_completion_tokens: int. Max tokens to generate when calling the Goodfire API.
          temperature: float for sampling temperature.
          kwargs: Additional arguments to store or pass into gen_kwargs.
        """
        super().__init__()
        self.api_key = api_key
        self.model_str = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self.kwargs = kwargs
        self._max_length = 4096

        try:
            self.client = goodfire.Client(api_key=self.api_key)
        except Exception as e:
            raise ValueError(
                f"Could not initialize Goodfire client with api_key={api_key}. Error: {e}"
            )

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self.max_completion_tokens

    def tok_encode(self, string: str, **kwargs) -> List[str]:
        """
        For chat-based models, we don't rely on tokens for loglikelihood. 
        Return the raw text as a single chunk. 
        """
        return [string]

    def tok_decode(self, tokens: List[str], **kwargs) -> str:
        return "".join(tokens)

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

    def loglikelihood(self, requests, **kwargs):
        raise NotImplementedError("GoodfireLLM does not support loglikelihood yet.")

    def loglikelihood_rolling(self, requests, **kwargs):
        raise NotImplementedError("GoodfireLLM does not support rolling loglikelihood.")

    def format_results(self, results):
        """Format results in a clean table format similar to VLLM output."""
        output = []
        
        # Add model info
        output.append(f"\nResults for Goodfire LLM Evaluation:")
        output.append("-" * 100)
        output.append(f"Model: goodfire (model={self.model_str})")
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

    def generate_until(self, requests, disable_tqdm=False) -> List[str]:
        """
        This is used for generation tasks. We'll call Goodfire's chat completions endpoint
        for each prompt individually and gather the results.
        """
        from tqdm import tqdm

        results = []
        for i, request in enumerate(tqdm(requests, disable=disable_tqdm)):
            prompt_str, gen_args = request.args
            until = gen_args.get("until") or []
            max_gen_toks = gen_args.get("max_gen_toks", self.max_completion_tokens)
            temperature = gen_args.get("temperature", self.temperature)
            top_p = gen_args.get("top_p", 1.0)  # Default to 1.0 if not specified

            # Debug: Log the prompt
            if i == 0:  # Log just the first prompt to avoid spam
                eval_logger.info("\nExample prompt:")
                eval_logger.info("-" * 50)
                eval_logger.info(prompt_str)
                eval_logger.info("-" * 50)

            messages = [{"role": "user", "content": prompt_str}]
            do_sample = gen_args.get("do_sample", True)

            gf_kwargs = {
                "model": self.model_str,
                "max_completion_tokens": max_gen_toks,
                "temperature": temperature,
                "top_p": top_p,
                "stream": False,
            }

            # Remove any None values to use API defaults
            gf_kwargs = {k: v for k, v in gf_kwargs.items() if v is not None}

            try:
                completion_response = self.client.chat.completions.create(messages, **gf_kwargs)

                if not completion_response or not completion_response.choices:
                    output_text = ""
                else:
                    output_text = completion_response.choices[0].message['content']

                # Debug: Log the response
                if i == 0:  # Log just the first response
                    eval_logger.info("\nExample response:")
                    eval_logger.info("-" * 50)
                    eval_logger.info(output_text)
                    eval_logger.info("-" * 50)

                for stop_seq in until:
                    pos = output_text.find(stop_seq)
                    if pos != -1:
                        output_text = output_text[:pos]

                results.append(output_text)
                self.cache_hook.add_partial("generate_until", request, output_text)
            except Exception as e:
                eval_logger.warning(f"Error in generate_until: {str(e)}")
                results.append("")  # Return empty string on error

        # If this was called through evaluate(), format the results
        if hasattr(self, '_current_task_list'):
            eval_logger.info("\nResults:")
            eval_logger.info("-" * 50)
            eval_logger.info(f"Model: goodfire (model={self.model_str})")
            eval_logger.info(f"Temperature: {self.temperature}")
            if top_p != 1.0:
                eval_logger.info(f"Top-p: {top_p}")
            eval_logger.info("-" * 50)

        return results