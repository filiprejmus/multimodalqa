from openai import AsyncOpenAI as OpenAI
from prompts import inferenceSystemPromptNaive
import json
import asyncio


class OpenAIInferer:
    def __init__(self):
        self.client = OpenAI()

    async def inference(self, textContext, imageContext, tableContext, question, qid):
        textSystemArray = []
        for text in textContext:
            textSystemArray.append({"type": "text", "text": text})
        textSystemMessage = {"role": "user", "content": textSystemArray}
        imageSystemArray = []
        for image in imageContext:
            imageSystemArray.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/{image['format']};base64,{image['image']}"
                    },
                }
            )
        imageSystemMessage = {"role": "user", "content": imageSystemArray}
        tableSystemMessage = {"role": "user", "content": tableContext}

        completion = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": inferenceSystemPromptNaive},
                textSystemMessage,
                imageSystemMessage,
                tableSystemMessage,
                {"role": "user", "content": question},
            ],
        )
        price_in_usd = (completion.usage.completion_tokens * 0.60 / 1000000) + (
            completion.usage.prompt_tokens * 0.15 / 1000000
        )
        return {
            "qid": qid,
            "result": json.loads(completion.choices[0].message.content),
            "price_in_usd": price_in_usd,
        }

    async def run_inferences(self, inferences):
        tasks = [self.inference(*inference) for inference in inferences]
        return await asyncio.gather(*tasks)
