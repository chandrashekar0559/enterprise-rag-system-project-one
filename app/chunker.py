def chunk_text(text, chunk_size=500, overlap=50):

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


if __name__ == "__main__":

    text = """
    Employees receive 20 days of annual leave per calendar year.
    Employees should submit leave requests through the internal HR portal.
    """

    chunks = chunk_text(text)

    for i, chunk in enumerate(chunks):

        print(f"\nChunk {i}:")
        print(chunk)