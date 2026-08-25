def render_rows(rows):
    lines = []
    for row in rows:
        cells = [str(cell).strip() for cell in row]
        joined = " | ".join(cells)
        lines.append(joined)
        lines.append("-" * len(joined))
    return "\n".join(lines)
