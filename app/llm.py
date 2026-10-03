from ollama import chat


class LLMClient:

    def __init__(self, model="llama3.2"):
        self.model = model

    def generate(self, prompt: str) -> str:

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.message.content