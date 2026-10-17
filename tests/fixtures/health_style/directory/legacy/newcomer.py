"""Helpers for the shelf records."""


def count_shelf_0(values, limit):
    """Return the count shelf 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 1 so the shelf totals line up.
    result = sum(kept) * 1
    if result > limit:
        return limit
    return result


def total_shelf_1(values, limit):
    """Return the total shelf 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the shelf totals line up.
    result = sum(kept) * 2
    return result


def label_shelf_2(values, limit):
    """Return the label shelf 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the shelf totals line up.
    result = sum(kept) * 3
    if result > limit:
        return limit
    return result


def check_shelf_3(values, limit):
    """Return the check shelf 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the shelf totals line up.
    result = sum(kept) * 4
    return result
