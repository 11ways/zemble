import logging
from dataclasses import replace

from zemble.chunking.capsule import CapsuleLevel, CapsuleOptions, FileContext, capsule
from zemble.chunking.core import chunk_lines, chunk_with_tree
from zemble.types import Chunk

logger = logging.getLogger(__name__)

# The desired length of chunks in chars.
# TODO: make this configurable
_DESIRED_CHUNK_LENGTH_CHARS = 750


def chunk_source(
    source: str,
    file_path: str,
    language: str | None,
    capsules: CapsuleOptions | None = None,
) -> list[Chunk]:
    """Chunk pre-read source text, stamping each chunk with its context capsule.

    :param source: The file's text.
    :param file_path: The repo-relative path stored on every chunk.
    :param language: The detected language, or None for line chunking.
    :param capsules: Capsule knobs; None resolves the environment override, else the defaults.
    :return: The file's chunks, in source order.
    """
    if not source.strip():
        return []
    level = CapsuleOptions.resolve(capsules).level
    chunk_boundaries = None
    root = None
    if language is not None:
        parsed = chunk_with_tree(source, language, _DESIRED_CHUNK_LENGTH_CHARS)
        if parsed is not None:
            chunk_boundaries, root = parsed
    # This is an if because the error state of the parser above
    # is a None.
    if chunk_boundaries is None:
        chunk_boundaries = chunk_lines(source, _DESIRED_CHUNK_LENGTH_CHARS)

    file_context = (
        FileContext(file_path=file_path, source=source, language=language, root=root)
        if level is not CapsuleLevel.OFF
        else None
    )

    chunks: list[Chunk] = []
    for boundary in chunk_boundaries:
        # Clamp to start_index so zero-length chunks don't produce an off-by-one.
        end_index = max(boundary.end - 1, boundary.start)
        text = source[boundary.start : end_index + 1]
        built = Chunk(
            content=text,
            file_path=file_path,
            start_line=source[: boundary.start].count("\n") + 1,
            end_line=source[:end_index].count("\n") + 1,
            language=language,
        )
        if file_context is not None:
            built = replace(built, context=capsule(built, file_context, level))
        chunks.append(built)
    return chunks
