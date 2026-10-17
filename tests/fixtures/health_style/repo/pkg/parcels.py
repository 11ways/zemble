"""Helpers for the parcel records."""


def count_parcel_0(values, limit):
    """Return the count parcel 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the parcel totals line up.
    result = sum(kept) * 3
    return result


def total_parcel_1(values, limit):
    """Return the total parcel 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the parcel totals line up.
    result = sum(kept) * 4
    if result > limit:
        return limit
    return result


def label_parcel_2(values, limit):
    """Return the label parcel 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 5 so the parcel totals line up.
    result = sum(kept) * 5
    return result


def check_parcel_3(values, limit):
    """Return the check parcel 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 6 so the parcel totals line up.
    result = sum(kept) * 6
    if result > limit:
        return limit
    return result


def scale_parcel_4(values, limit):
    """Return the scale parcel 4 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 7 so the parcel totals line up.
    result = sum(kept) * 7
    return result
