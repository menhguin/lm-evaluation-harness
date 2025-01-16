import json
import re
from typing import Union, Dict, List, Optional
import os

# Move goodfire import inside try block
try:
    import goodfire
    _has_goodfire = True
except ImportError:
    _has_goodfire = False

from lm_eval import utils
from lm_eval.api.model import LM
from lm_eval.api.instance import Instance
from lm_eval.models.utils import handle_stop_sequences
from lm_eval.api.registry import register_model
from tqdm import tqdm

eval_logger = utils.eval_logger

def get_goodfire_api_key() -> str:
    """Get Goodfire API key from environment."""
    return os.getenv('GOODFIRE_API_KEY')

def _debug_log_prompt(prompt: str, index: int) -> None:
    """Log a prompt for debugging."""
    eval_logger.info(f"\n{'='*50}\nPROMPT #{index + 1}:\n{'='*50}\n{prompt}\n{'='*50}\n")

def _debug_log_response(response: str, index: int) -> None:
    """Log a response for debugging."""
    eval_logger.info(f"\n{'='*50}\nRESPONSE #{index + 1}:\n{'='*50}\n{response}\n{'='*50}\n")

def _debug_log_processed(processed: str, index: int, stop_seq: str = None) -> None:
    """Log processed output for debugging."""
    eval_logger.info(f"\n{'='*50}\nPROCESSED #{index + 1}:\n{'='*50}\n{processed}\n")
    if stop_seq:
        eval_logger.info(f"(Truncated at stop sequence: {stop_seq})\n{'='*50}\n")
    else:
        eval_logger.info(f"(No truncation)\n{'='*50}\n")

@register_model("goodfire")
class GoodfireLLM(LM):
    """
    Integration of Goodfire's chat API with the EleutherAI lm-eval-harness.
    This class uses the Goodfire client for completions.
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
        if not _has_goodfire:
            raise ModuleNotFoundError(
                'Package `goodfire` not found. Please install it via `pip install goodfire`'
            )
        super().__init__()
        self.api_key = api_key or get_goodfire_api_key()
        if not self.api_key:
            raise ValueError("Goodfire API key is required but not found. Please set GOODFIRE_API_KEY in environment.")
        
        self.client = goodfire.Client(api_key=self.api_key)
        self.model = model
        self.max_completion_tokens = max_completion_tokens
        self.temperature = temperature
        self._max_length = 4096  # Default max length for context + completion

    def _generate_completion(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float,
        top_p: float,
        idx: int = 0
    ) -> str:
        """Generate a single completion."""
        try:
            # For chat history, combine messages with newlines to maintain format
            if len(messages) > 1:
                combined_message = "\n".join(msg["content"] for msg in messages)
                messages = [{"role": "user", "content": combined_message}]
            
            api_params = {
                "messages": messages,
                "model": self.model,
                "max_completion_tokens": self.max_completion_tokens,
                "temperature": temperature,
                "top_p": top_p
            }
            
            response = self.client.chat.completions.create(**api_params)
            output = response.choices[0].message['content']
            _debug_log_response(output, idx)
            return output
        except Exception as e:
            eval_logger.error(f"Error generating response #{idx + 1}: {str(e)}")
            return ""

    def generate_until(
        self, requests: List[Instance], disable_tqdm: bool = False
    ) -> List[str]:
        """Generate responses for a list of requests."""
        self._is_generating = True
        try:
            results = []
            with tqdm(total=len(requests), disable=disable_tqdm, desc="Running requests") as pbar:
                for idx, req in enumerate(requests):
                    context, gen_kwargs = req.args
                    
                    # Extract generation parameters
                    if isinstance(gen_kwargs, dict):
                        kwargs = gen_kwargs.copy()
                        until = handle_stop_sequences(kwargs.pop("until", []), eos=None)
                        do_sample = kwargs.pop("do_sample", True)
                        temperature = kwargs.pop("temperature", self.temperature)
                        if not do_sample:
                            temperature = 0.0
                        top_p = kwargs.pop("top_p", 1.0)
                    else:
                        until = []
                        temperature = self.temperature
                        top_p = 1.0

                    # Log prompt
                    _debug_log_prompt(str(context), idx)

                    # Prepare messages - just pass through as single user message
                    messages = [{"role": "user", "content": context}]

                    # Generate completion
                    output = self._generate_completion(
                        messages=messages,
                        temperature=temperature,
                        top_p=top_p,
                        idx=idx
                    )

                    # Process stop sequences
                    if until:
                        for stop_seq in until:
                            if stop_seq in output:
                                output = output[:output.index(stop_seq)].strip()
                                _debug_log_processed(output, idx, stop_seq)
                                break
                        else:
                            _debug_log_processed(output, idx)
                    else:
                        _debug_log_processed(output, idx)

                    results.append(output)
                    pbar.update(1)

            return results
        finally:
            self._is_generating = False

    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """Create an instance from an argument string."""
        args = utils.simple_parse_args_string(arg_string)
        pretrained = args.pop("pretrained", "meta-llama/Meta-Llama-3.1-8B-Instruct")
        
        # Filter out irrelevant args for API-only implementation
        api_relevant_args = {
            k: v for k, v in args.items() 
            if k in ["api_key", "max_completion_tokens", "temperature"]
        }
        
        return cls(model=pretrained, **api_relevant_args)

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

    def apply_chat_template(
        self, messages: List[Dict[str, str]], system_message: str = None
    ) -> Union[str, List[Dict[str, str]]]:
        """Apply chat template to format messages for the model.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content'
            system_message: Optional system message to prepend
        
        Returns:
            Either a string representation for hashing or the formatted messages for the API
        """
        formatted_messages = []
        
        # Add system message if provided
        if system_message:
            formatted_messages.append({
                "role": "system",
                "content": system_message
            })
        
        # Add all other messages
        formatted_messages.extend(messages)
        
        # For generate_until, return the messages list
        if hasattr(self, '_is_generating') and self._is_generating:
            return formatted_messages
            
        # For hashing and other purposes, return a string representation
        return json.dumps(formatted_messages, sort_keys=True)