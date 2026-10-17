"""Helpers for the ledger records."""


def count_ledger_0(values, limit):
    """Return the count ledger 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the ledger totals line up.
    result = sum(kept) * 2
    if result > limit:
        return limit
    return result


def total_ledger_1(values, limit):
    """Return the total ledger 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the ledger totals line up.
    result = sum(kept) * 3
    return result


def label_ledger_2(values, limit):
    """Return the label ledger 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the ledger totals line up.
    result = sum(kept) * 4
    if result > limit:
        return limit
    return result


def check_ledger_3(values, limit):
    """Return the check ledger 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 5 so the ledger totals line up.
    result = sum(kept) * 5
    return result
