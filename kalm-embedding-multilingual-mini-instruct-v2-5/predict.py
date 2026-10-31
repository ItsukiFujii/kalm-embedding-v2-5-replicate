"""Cog interface for KaLM-Embedding-V2.5."""

from copy import copy
import os
from pathlib import Path

from cog import BaseRunner, Input
from typing_extensions import TypedDict


class EmbeddingResult(TypedDict):
    index: int
    embedding: list[float]


class Runner(BaseRunner):
    def setup(self) -> None:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

        import torch
        from sentence_transformers import SentenceTransformer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = (
            torch.bfloat16
            if device == "cuda" and torch.cuda.is_bf16_supported()
            else torch.float32
        )
        self.model = SentenceTransformer(
            str(Path(__file__).resolve().parent / "weights"),
            device=device,
            trust_remote_code=True,
            local_files_only=True,
            model_kwargs={"torch_dtype": dtype, "attn_implementation": "sdpa"},
        )
        self.model.max_seq_length = 512
        self.model.default_prompt_name = None
        self.model.eval()
        self.model.requires_grad_(False)

    def run(
        self,
        texts: list[str] = Input(
            description="Array of texts to embed. Returned vectors follow the input order."
        ),
        instruction: str = Input(
            default="",
            description=(
                "Optional task description. Leave empty to encode the original texts. "
                "When provided, each input becomes Instruct: {instruction} followed by "
                "a newline and Query:{text}. Enter only the task description, without template labels."
            ),
        ),
        embedding_dim: int = Input(
            default=896,
            choices=[896, 512, 256, 128, 64],
            description="Output embedding dimension supported by Matryoshka representation learning.",
        ),
        batch_size: int = Input(
            default=32,
            ge=1,
            description="Number of texts processed together in each inference batch.",
        ),
        max_length: int = Input(
            default=512,
            ge=1,
            le=32768,
            description="Maximum number of tokens per formatted text, including the optional instruction. Longer inputs are truncated.",
        ),
        normalize: bool = Input(
            default=True,
            description="Whether to L2-normalize each embedding to unit length.",
        ),
    ) -> list[EmbeddingResult]:
        if not isinstance(texts, list):
            raise ValueError("texts must be an array of strings.")
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise ValueError(f"texts[{index}] must be a non-empty string.")
        if not isinstance(instruction, str):
            raise ValueError("instruction must be a string.")
        if type(embedding_dim) is not int or embedding_dim not in (896, 512, 256, 128, 64):
            raise ValueError("embedding_dim must be one of 896, 512, 256, 128, or 64.")
        if type(batch_size) is not int or batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")
        if type(max_length) is not int or not 1 <= max_length <= 32768:
            raise ValueError("max_length must be an integer from 1 to 32768.")
        if type(normalize) is not bool:
            raise ValueError("normalize must be a boolean.")
        if not texts:
            return []

        import torch
        from sentence_transformers.models import Normalize

        # An explicit empty prompt disables all saved prompt presets.
        prompt = f"Instruct: {instruction}\nQuery:" if instruction else ""
        # Request-local wrappers share the loaded weights and tokenizer. Their
        # dimension, sequence limit and module list never mutate the base model.
        request_model = copy(self.model)
        request_model._modules = self.model._modules.copy()
        first_name = next(iter(request_model._modules))
        request_transformer = copy(request_model._modules[first_name])
        request_transformer.max_seq_length = max_length
        request_model._modules[first_name] = request_transformer
        request_model.truncate_dim = embedding_dim
        if not normalize:
            # The exported model already contains a Normalize module. Bypass it
            # as well as encode() normalization to return raw pooled embeddings.
            request_model._modules = {
                name: module
                for name, module in request_model._modules.items()
                if not isinstance(module, Normalize)
            }
        with torch.inference_mode():
            embeddings = request_model.encode(
                texts,
                prompt=prompt,
                batch_size=batch_size,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                convert_to_tensor=True,
            ).float().cpu()
        if embeddings.shape != (len(texts), embedding_dim):
            raise RuntimeError("The model returned an unexpected embedding shape.")
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("The model returned non-finite embedding values.")
        return [
            {"index": index, "embedding": vector}
            for index, vector in enumerate(embeddings.tolist())
        ]
