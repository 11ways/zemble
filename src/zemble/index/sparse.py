from pathlib import Path

import numpy as np
import numpy.typing as npt

from zemble.chunking.capsule import capsule_without_path
from zemble.types import Chunk


def selector_to_mask(selector: npt.NDArray[np.int_] | None, size: int) -> npt.NDArray[np.bool_] | None:
    """Convert a selector array of indices into a boolean mask of length ``size``."""
    if selector is None:
        return None
    mask = np.zeros(size, dtype=bool)
    mask[selector] = True
    return mask


def enrich_for_bm25(chunk: Chunk, include_context: bool = False) -> str:
    """Append file path components to BM25 content to boost path-based queries.

    Assumes ``chunk.file_path`` is already repo-relative (set by ``create_index_from_path``)
    so machine-specific directory components are never indexed.

    :param chunk: The chunk to build a BM25 document for.
    :param include_context: Also append the chunk's capsule, minus its leading path
        segment, whose tokens this function already contributes.
    """
    path = Path(chunk.file_path)
    stem = path.stem
    dir_parts = [part for part in path.parent.parts if part not in (".", "/")]
    dir_text = " ".join(dir_parts[-3:])  # Last 3 directory components
    # Repeat the stem twice to up-weight file-path matches in BM25.
    enriched = f"{chunk.content} {stem} {stem} {dir_text}"
    if include_context and (context := capsule_without_path(chunk.context)):
        enriched = f"{enriched} {context}"
    return enriched
