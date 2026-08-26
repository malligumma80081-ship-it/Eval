import faiss
import numpy as np


class VectorStore:

    def __init__(self, embeddings):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(
            dimension
        )

        self.index.add(
            embeddings.astype(
                np.float32
            )
        )


    def search(
        self,
        query_embedding,
        top_k=3
    ):

        distances, indices = (
            self.index.search(
                query_embedding.astype(
                    np.float32
                ),
                top_k
            )
        )

        return distances, indices