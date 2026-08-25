def render_table(records):
    output = []
    for record in records:
        parts = [str(part).strip() for part in record]
        text = " | ".join(parts)
        output.append(text)
        output.append("-" * len(text))
    return "\n".join(output)
