"""A dependency-free token estimate for budgeting evidence bundles."""

from __future__ import annotations

# Characters per token, measured over this workspace's Java, Python and Markdown
# source with a real BPE tokenizer (tiktoken `cl100k_base`): code sits near 3.6,
# prose near 4.0. The lower of the two is used so a budget is met, not missed.
CHARS_PER_TOKEN = 3.6


def estimate_tokens(text: str) -> int:
    """Estimate the token count of a string.

    This is an ESTIMATE, not a tokenizer: it divides the character count by
    `CHARS_PER_TOKEN`. It never imports a model and never touches the network,
    which is why the packer can call it thousands of times per bundle. Expect it
    to be within roughly 10 percent of a BPE count on source text, and to be
    worse on text that is mostly punctuation or non-ASCII.

    :param text: The text to measure.
    :return: The estimated number of tokens, at least 1 for non-empty text.
    """
    if not text:
        return 0
    return max(1, round(len(text) / CHARS_PER_TOKEN))
