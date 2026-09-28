from .prompt_template import RAG_SYSTEM_PROMPT, build_rag_prompt
from .cache import SemanticCache


def __getattr__(name):
    if name == "RAGPipeline":
        from .rag_pipeline import RAGPipeline
        return RAGPipeline
    raise AttributeError(f"module 'generation' has no attribute '{name}'")


__all__ = [
    "RAG_SYSTEM_PROMPT", "build_rag_prompt",
    "RAGPipeline", "SemanticCache",
]
