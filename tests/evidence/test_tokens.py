"""The token estimate is cheap, monotonic and roughly right."""

from zemble.evidence.tokens import CHARS_PER_TOKEN, estimate_tokens


def test_estimate_journey() -> None:
    """Walk the estimate through the cases the packer relies on."""
    # 1. Nothing costs nothing.
    assert estimate_tokens("") == 0, "step 1: empty text is free"

    # 2. Any non-empty text costs at least one token.
    assert estimate_tokens("x") == 1, "step 2: a single character still costs a token"

    # 3. The estimate is the character count over the documented ratio.
    text = "public static int add(int a, int b) { return a + b; }"
    assert estimate_tokens(text) == round(len(text) / CHARS_PER_TOKEN), "step 3: the ratio is the whole rule"

    # 4. More text never costs less.
    assert estimate_tokens(text * 3) > estimate_tokens(text), "step 4: the estimate is monotonic"
