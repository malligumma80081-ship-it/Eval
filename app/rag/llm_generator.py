import ollama


class LlamaGenerator:

    def __init__(
        self,
        model_name="llama3.2"
    ):
        self.model_name = model_name

    def generate(
        self,
        question,
        contexts
    ):

        context_text = "\n\n".join(
            context["text"]
            for context in contexts
        )

        prompt = f"""
You are a question answering system.

Answer the user's question using ONLY
the provided context.

If the answer is not available in the
context, say:

"I don't know based on the provided context."

Context:
{context_text}

Question:
{question}

Answer:
"""

        response = ollama.chat(
            model=self.model_name,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"].strip()