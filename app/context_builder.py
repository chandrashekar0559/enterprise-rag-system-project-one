def build_context(results):

    context_parts = []

    for result in results:

        text = result["text"]
        metadata = result["metadata"]

        source = metadata.get(
            "source",
            "Unknown source"
        )

        context_parts.append(
            f"[Source: {source}]\n{text}"
        )

    return "\n\n".join(context_parts)