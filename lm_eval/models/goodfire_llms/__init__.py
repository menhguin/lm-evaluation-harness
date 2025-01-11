from .model import GoodfireLLM
from lm_eval.api.registry import register_model

register_model("goodfire_llms", GoodfireLLM)

__all__ = ["GoodfireLLM"] 