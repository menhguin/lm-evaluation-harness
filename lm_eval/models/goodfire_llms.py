import json
from typing import Union, Dict, List, Optional
import asyncio

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

    def generate_until(self, requests, disable_tqdm=False) -> List[str]:
        """
        This is used for generation tasks. We'll call Goodfire's chat completions endpoint
        for each prompt individually and gather the results.
        """
        from tqdm import tqdm

        results = []
        for request in tqdm(requests, disable=disable_tqdm):
            prompt_str, gen_args = request.args
            until = gen_args.get("until") or []
            max_gen_toks = gen_args.get("max_gen_toks", self.max_completion_tokens)
            temperature = gen_args.get("temperature", self.temperature)

            messages = [{"role": "user", "content": prompt_str}]
            do_sample = gen_args.get("do_sample", True)

            gf_kwargs = {
                "model": self.model_str,
                "max_completion_tokens": max_gen_toks,
                "temperature": temperature,
                "stream": False,
            }

            completion_response = self.client.chat.completions.create(messages, **gf_kwargs)

            if not completion_response or not completion_response.choices:
                output_text = ""
            else:
                output_text = completion_response.choices[0].message['content']

            for stop_seq in until:
                pos = output_text.find(stop_seq)
                if pos != -1:
                    output_text = output_text[:pos]

            results.append(output_text)
            self.cache_hook.add_partial("generate_until", request, output_text)

        return results 