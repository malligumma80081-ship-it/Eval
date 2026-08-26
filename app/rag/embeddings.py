from sentence_transformers import SentenceTransformer


def chunk_text(
    text,
    chunk_size=300
):

    chunks = []

    start = 0

    while start < len(text):

        chunk = text[
            start:start + chunk_size
        ]

        chunks.append(chunk)

        start += chunk_size

    return chunks



model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


def create_embeddings(texts):

    embeddings = model.encode(
        texts,
        convert_to_numpy=True
    )

    return embeddings