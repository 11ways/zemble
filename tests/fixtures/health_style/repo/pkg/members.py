"""Helpers for the member records."""


def count_member_0(values, limit):
    """Return the count member 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the member totals line up.
    result = sum(kept) * 2
    return result


def total_member_1(values, limit):
    """Return the total member 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the member totals line up.
    result = sum(kept) * 3
    if result > limit:
        return limit
    return result


def label_member_2(values, limit):
    """Return the label member 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the member totals line up.
    result = sum(kept) * 4
    return result


def check_member_3(values, limit):
    """Return the check member 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 5 so the member totals line up.
    result = sum(kept) * 5
    if result > limit:
        return limit
    return result
