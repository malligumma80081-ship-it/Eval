from app.rag.document_loader import (
    load_documents
)

from app.rag.embeddings import (
    chunk_text,
    create_embeddings
)

from app.rag.vector_store import (
    VectorStore
)


class RAGRetriever:

    def __init__(
        self,
        document_folder: str
    ):

        self.documents = load_documents(
            document_folder
        )

        self.chunks = []

        self.build_index()


    def build_index(self):

        for document in self.documents:

            source = document["source"]

            chunks = chunk_text(
                document["text"]
            )

            for chunk_number, chunk in enumerate(
                chunks
            ):

                self.chunks.append(
                    {
                        "chunk_id": (
                            f"{source}:chunk_{chunk_number}"
                        ),
                        "source": source,
                        "text": chunk
                    }
                )

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        embeddings = create_embeddings(
            texts
        )

        self.vector_store = VectorStore(
            embeddings
        )


    def retrieve(
        self,
        question: str,
        top_k: int = 10
    ):

        query_embedding = create_embeddings(
            [question]
        )

        scores, indices = (
            self.vector_store.search(
                query_embedding,
                top_k
            )
        )

        results = []

        for score, index in zip(
            scores[0],
            indices[0]
        ):

            chunk = self.chunks[index]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "source": chunk["source"],
                    "text": chunk["text"],
                    "score": float(score)
                }
            )

        return results