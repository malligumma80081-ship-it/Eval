# import os

# import ollama


# class OllamaClient:

#     def __init__(self, model=None, host=None, timeout=None):
#         self.model = model or os.getenv("OLLAMA_MODEL")
#         self.host = host or os.getenv("OLLAMA_HOST")
#         self.timeout = timeout if timeout is not None else int(os.getenv("OLLAMA_TIMEOUT", "20"))

#     def generate(self, prompt):
#         if not self.model:
#             raise ValueError(
#                 "No Ollama model configured. Pass model='...' to OllamaClient() or set the OLLAMA_MODEL environment variable."
#             )

#         try:
#             client = ollama.Client(host=self.host, timeout=self.timeout) if self.host else ollama.Client(timeout=self.timeout)
#             response = client.chat(
#                 model=self.model,
#                 messages=[
#                     {
#                         "role": "user",
#                         "content": prompt
#                     }
#                 ],
#                 stream=False,
#             )
#         except Exception as exc:
#             raise RuntimeError(
#                 "Ollama is not available or not running. Start the Ollama service and try again."
#             ) from exc

#         return response["message"]["content"]


# if __name__ == "__main__":

#     client = OllamaClient()

#     answer = client.generate(
#         "Explain Python in one sentence."
#     )

#     print(answer)


import os

import ollama


class OllamaClient:

    def __init__(self, model=None, host=None, timeout=None):
        self.model = model or os.getenv("OLLAMA_MODEL") or "llama3.2"
        self.host = host or os.getenv("OLLAMA_HOST")
        self.timeout = timeout if timeout is not None else int(os.getenv("OLLAMA_TIMEOUT", "5"))

    def generate(self, prompt):
        if not self.model:
            raise ValueError(
                "No Ollama model configured. Pass model='...' to OllamaClient() or set the OLLAMA_MODEL environment variable."
            )

        try:
            client = ollama.Client(host=self.host, timeout=self.timeout) if self.host else ollama.Client(timeout=self.timeout)
            response = client.chat(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                stream=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Ollama is not available or not running. Start the Ollama service and try again."
            ) from exc

        return response["message"]["content"]


if __name__ == "__main__":

    client = OllamaClient()

    answer = client.generate(
        "Explain Python in one sentence."
    )

    print(answer)