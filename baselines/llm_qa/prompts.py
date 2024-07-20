inferenceSystemPromptNaive = """
You are getting a request to answer a question that requires combined information from a table,
multiple text paragraphs and multiple images. Some of the information is relevant to the question,
while other information is not.

Use JSON mode and answer the question in this format:
{{
    "answers": ["answer1", "answer2"]
}}

Naturally there can be either one or more answers.
Your answers should always represent a single entity. In other words if the question is about a year or a number,
just include the number. If the answer is a name or title include the full name or title. If the answer is about an adjective, 
just include the adjective and so on.

"""
