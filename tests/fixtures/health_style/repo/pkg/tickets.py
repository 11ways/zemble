"""Helpers for the ticket records."""


def count_ticket_0(values, limit):
    """Return the count ticket 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 1 so the ticket totals line up.
    result = sum(kept) * 1
    if result > limit:
        return limit
    return result


def total_ticket_1(values, limit):
    """Return the total ticket 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the ticket totals line up.
    result = sum(kept) * 2
    return result


def label_ticket_2(values, limit):
    """Return the label ticket 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the ticket totals line up.
    result = sum(kept) * 3
    if result > limit:
        return limit
    return result
