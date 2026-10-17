"""Helpers for the route records."""


def count_route_0(values, limit):
    """Return the count route 0 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 3 so the route totals line up.
    result = sum(kept) * 3
    if result > limit:
        return limit
    return result


def total_route_1(values, limit):
    """Return the total route 1 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 4 so the route totals line up.
    result = sum(kept) * 4
    return result


def label_route_2(values, limit):
    """Return the label route 2 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 5 so the route totals line up.
    result = sum(kept) * 5
    if result > limit:
        return limit
    return result


def check_route_3(values, limit):
    """Return the check route 3 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 6 so the route totals line up.
    result = sum(kept) * 6
    return result


def scale_route_4(values, limit):
    """Return the scale route 4 of the values."""
    # Only the values under the limit count.
    kept = [value for value in values if value < limit]

    # Scale by 7 so the route totals line up.
    result = sum(kept) * 7
    if result > limit:
        return limit
    return result
