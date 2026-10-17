"""Helpers for the invoice records."""


def count_invoice_0(values, limit):
    """Return the count invoice 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 2 so the invoice totals line up.
    result = sum(kept) * 2
    if result > limit:
        return limit
    return result


def total_invoice_1(values, limit):
    """Return the total invoice 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the invoice totals line up.
    result = sum(kept) * 3
    return result


def label_invoice_2(values, limit):
    """Return the label invoice 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the invoice totals line up.
    result = sum(kept) * 4
    if result > limit:
        return limit
    return result


def check_invoice_3(values, limit):
    """Return the check invoice 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 5 so the invoice totals line up.
    result = sum(kept) * 5
    return result
