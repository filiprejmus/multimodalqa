from openai import OpenAI
from prompts import inferenceSystemPromptNaive


class OpenAIInferer:
    def __init__(self):
        self.client = OpenAI()

    def inference(self):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": inferenceSystemPromptNaive},
                {"role": "user", "content": "Hello!"},
            ],
        )
        return completion.choices[0].message["content"]
