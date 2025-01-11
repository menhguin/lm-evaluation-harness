import json
from typing import Union, Dict, List, Optional, Any
import os

try:
    import goodfire
    from google.colab import userdata  # For Colab integration
except ImportError as e:
    if "google.colab" in str(e):
        userdata = None  # Not running in Colab
    else:
        raise ModuleNotFoundError(
            "Please install goodfire to use Goodfire models: pip install goodfire"
        ) from e

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.api.registry import register_model
from tqdm import tqdm


eval_logger = utils.eval_logger


def get_goodfire_api_key() -> str:
    """Get Goodfire API key from environment or Colab secrets."""
    api_key = os.getenv('GOODFIRE_API_KEY')
    if not api_key and userdata is not None:
        # Try Colab secrets if available
        api_key = userdata.get('GOODFIRE_API_KEY')
        if api_key:
            # Set in environment for future use
            os.environ['GOODFIRE_API_KEY'] = api_key
    return api_key


@register_model("goodfire")
class GoodfireLLM(LM):
    """
    Integration of Goodfire's chat API with the EleutherAI lm-eval-harness.
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
            api_key: str. Your Goodfire API key. If not provided, tries environment variable.
            model: str. The Goodfire model to use, e.g. "meta-llama/Meta-Llama-3-8B-Instruct".
            max_completion_tokens: int. Max tokens to generate when calling the Goodfire API.
            temperature: float. Sampling temperature.
        """
        super().__init__()
        self.api_key = api_key or get_goodfire_api_key()
        if not self.api_key:
            raise ValueError("Goodfire API key is required but not found. Please set GOODFIRE_API_KEY in environment or Colab secrets.")
        
        self.client = goodfire.Client(api_key=self.api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._max_length = 4096  # Default max length for context + completion

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """Create an instance from an argument string."""
        args = utils.simple_parse_args_string(arg_string)
        pretrained = args.pop("pretrained", "meta-llama/Meta-Llama-3-8B-Instruct")
        return cls(model=pretrained, **args)

    @property
    def max_length(self) -> int:
        return self._max_length

    @property
    def max_gen_toks(self) -> int:
        return self.max_completion_tokens

    @property
    def tokenizer_name(self) -> str:
        """Return the name of the tokenizer to use for chat templates."""
        # For Llama models, use the Llama tokenizer
        if "llama" in self.model.lower():
            return "meta-llama/Llama-2-7b-hf"
        # Default to Llama tokenizer if unknown
        return "meta-llama/Llama-2-7b-hf"

    def tok_encode(self, string: str) -> List[int]:
        """Placeholder for tokenization - actual token count not needed."""
        return [0] * (len(string) // 4)  # Rough estimate

    def tok_decode(self, tokens: List[int]) -> str:
        """Not needed for generation."""
        return ""

    def _model_call(self, inps):
        """Not used by lm-eval-harness for generate_until."""
        raise NotImplementedError("GoodfireLLM does not support direct _model_call usage.")

    def _model_generate(self, context, max_length, eos_token_id):
        """Not used, we override 'generate_until'."""
        raise NotImplementedError("GoodfireLLM does not use _model_generate in this integration.")

    def loglikelihood(self, requests):
        raise NotImplementedError("Loglikelihood not supported for Goodfire models")

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError("Loglikelihood_rolling not supported for Goodfire models")

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """Generate responses for a list of requests."""
        res = []
        pbar = tqdm(total=len(requests), disable=disable_tqdm, desc="Running requests")

        for req in requests:
            context, gen_kwargs = req.args
            
            # Extract generation parameters from task config
            if isinstance(gen_kwargs, dict):
                kwargs = gen_kwargs.copy()
                # Get stop sequences from task config
                until = handle_stop_sequences(kwargs.pop("until", []), eos=None)
                # Get other generation parameters
                do_sample = kwargs.pop("do_sample", True)
                temperature = kwargs.pop("temperature", 1.0 if do_sample else 0.0)
                top_p = kwargs.pop("top_p", 1.0)
            else:
                until = []
                do_sample = True
                temperature = self.temperature
                top_p = 1.0

            # Log the first prompt for debugging
            if len(res) == 0:
                eval_logger.debug(f"\nFirst prompt: {context}\n")

            try:
                response = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": context}],
                    model=self.model,
                    max_completion_tokens=self.max_completion_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample
                )
                
                # Extract content from ChatCompletion object
                output = response.choices[0].message['content']
                
                # Log the first response for debugging
                if len(res) == 0:
                    eval_logger.debug(f"\nFirst response: {output}\n")
                
                # Handle stop sequences if provided
                if until:
                    # Try each stop sequence
                    for stop_seq in until:
                        if stop_seq in output:
                            output = output[:output.index(stop_seq)]
                            break  # Stop at first matching sequence

                res.append(output)
                pbar.update(1)
            except Exception as e:
                eval_logger.warning(f"Error generating response: {str(e)}")
                # Print more details about the response for debugging
                if 'response' in locals():
                    eval_logger.debug(f"Response type: {type(response)}")
                    eval_logger.debug(f"Response content: {response}")
                res.append("")  # Return empty string on error
                pbar.update(1)

        pbar.close()
        return res
```