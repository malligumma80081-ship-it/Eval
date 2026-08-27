import ollama

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


class RAGPipeline:

    def __init__(
        self,
        document_folder,
        model="llama3.2"
    ):

        self.model = model

        self.documents = load_documents(
            document_folder
        )

        self.chunks = []

        self.sources = []

        self.build_index()


    def build_index(self):

        for document in self.documents:

            chunks = chunk_text(
                document["text"]
            )

            for chunk in chunks:

                self.chunks.append(
                    chunk
                )

                self.sources.append(
                    document["source"]
                )

        embeddings = create_embeddings(
            self.chunks
        )

        self.vector_store = VectorStore(
            embeddings
        )


    def retrieve(
        self,
        question,
        top_k=10
    ):

        query_embedding = create_embeddings(
            [question]
        )

        distances, indices = (
            self.vector_store.search(
                query_embedding,
                top_k
            )
        )

        retrieved_chunks = []

        for index in indices[0]:

            retrieved_chunks.append({
                "text": self.chunks[index],
                "source": self.sources[index]
            })

        return retrieved_chunks


    def generate(
        self,
        question,
        retrieved_chunks
    ):

        context = "\n\n".join(
            item["text"]
            for item in retrieved_chunks
        )

        prompt = f"""
You are a helpful assistant.

Answer the question using only
the provided context.

Context:
{context}

Question:
{question}

If the answer is not available
in the context, say:
"I don't know based on the provided context."

Answer:
"""

        response = ollama.chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]


    def ask(
        self,
        question,
        top_k=10
    ):

        retrieved_chunks = self.retrieve(
            question,
            top_k
        )

        answer = self.generate(
            question,
            retrieved_chunks
        )

        return {
            "question": question,
            "context": retrieved_chunks,
            "answer": answer}