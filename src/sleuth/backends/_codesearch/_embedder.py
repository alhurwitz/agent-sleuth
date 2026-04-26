"""Optional cosine re-rank for CodeSearch.

When ``rerank=False`` (default), all methods are no-ops and fastembed is
never imported.  Set ``rerank=True`` only when the ``code-embed`` extra is
installed.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

from sleuth.types import Chunk

logger = logging.getLogger("sleuth.backends.codesearch.embedder")

_DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"


class Embedder:
    """Wraps fastembed for optional cosine re-rank of code chunks."""

    def __init__(
        self,
        *,
        rerank: bool = False,
        model_name: str = _DEFAULT_MODEL,
    ) -> None:
        self._rerank = rerank
        self._model_name = model_name
        self._model = None  # lazy-initialized

    def _load_model(self) -> None:
        if self._model is not None:
            return
        try:
            from fastembed import TextEmbedding  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(
                "Embedding re-rank requires 'agent-sleuth[code-embed]'. "
                "Install with: uv add agent-sleuth[code-embed]"
            ) from exc
        self._model = TextEmbedding(model_name=self._model_name)
        logger.debug("Loaded fastembed model: %s", self._model_name)

    async def rerank(self, query: str, chunks: list[Chunk]) -> list[Chunk]:
        """Return *chunks* reordered by cosine similarity to *query*.

        If ``self._rerank`` is False, returns *chunks* unchanged (fast path).
        """
        if not self._rerank or not chunks:
            return chunks

        import asyncio

        import numpy as np  # type: ignore[import-not-found]

        self._load_model()

        # fastembed is sync — run in executor to avoid blocking the event loop
        loop = asyncio.get_running_loop()

        def _embed_all() -> tuple[list[list[float]], list[list[float]]]:
            assert self._model is not None
            q_emb = list(self._model.embed([query]))
            c_embs = list(self._model.embed([c.text for c in chunks]))
            return q_emb, c_embs

        q_emb_list, c_embs_list = await loop.run_in_executor(None, _embed_all)

        q_vec = np.array(q_emb_list[0])
        c_vecs = np.array(c_embs_list)

        # Cosine similarity: q·c / (||q|| * ||c||)
        norms_c = np.linalg.norm(c_vecs, axis=1, keepdims=True)
        norms_c = np.where(norms_c == 0, 1.0, norms_c)
        c_vecs_norm = c_vecs / norms_c
        q_norm = q_vec / (np.linalg.norm(q_vec) or 1.0)
        scores = c_vecs_norm @ q_norm

        ranked_indices = np.argsort(scores)[::-1].tolist()
        return [chunks[i] for i in ranked_indices]
