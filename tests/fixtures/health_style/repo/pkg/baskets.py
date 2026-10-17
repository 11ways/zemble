"""Helpers for the basket records."""


def count_basket_0(values, limit):
    """Return the count basket 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 1 so the basket totals line up.
    result = sum(kept) * 1
    return result


def total_basket_1(values, limit):
    """Return the total basket 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the basket totals line up.
    result = sum(kept) * 2
    if result > limit:
        return limit
    return result


def label_basket_2(values, limit):
    """Return the label basket 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the basket totals line up.
    result = sum(kept) * 3
    return result
