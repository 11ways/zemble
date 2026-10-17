"""Helpers for the order records."""


def count_order_0(values, limit):
    """Return the count order 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 1 so the order totals line up.
    result = sum(kept) * 1
    return result


def total_order_1(values, limit):
    """Return the total order 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the order totals line up.
    result = sum(kept) * 2
    if result > limit:
        return limit
    return result


def label_order_2(values, limit):
    """Return the label order 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the order totals line up.
    result = sum(kept) * 3
    return result
