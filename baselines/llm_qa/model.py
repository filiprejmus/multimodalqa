from openai import AsyncOpenAI as OpenAI
from prompts import inferenceSystemPromptNaive
import json
import asyncio
import logging

logging.basicConfig(
    filename="inference_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


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
                    "image_url": {"url": f"data:image/png;base64,{image['image']}"},
                }
            )
        imageSystemMessage = {"role": "user", "content": imageSystemArray}
        tableSystemMessage = {"role": "user", "content": tableContext}

        try:
            completion = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                temperature=0.0,
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
            try:
                result = json.loads(completion.choices[0].message.content)
            except Exception as e:
                logging.error(
                    "Error processing completion for qid %s: %s, completion: %s",
                    qid,
                    str(e),
                    completion,
                )
                result = {"answers": []}
        except Exception as e:
            logging.error(
                "Error processing completion for qid %s: %s",
                qid,
                str(e),
            )
            result = {"answers": []}
            price_in_usd = 0.0
        return {
            "qid": qid,
            "result": result,
            "price_in_usd": price_in_usd,
        }

    async def run_inferences(self, inferences):
        tasks = [self.inference(*inference) for inference in inferences]
        return await asyncio.gather(*tasks)
