'''
Author: Armand Meijers
Date: 12/04/2026
Description: Generates answers from retrieved chunks using claude
'''

import anthropic

client = anthropic.Anthropic()  # automatically reads ANTHROPIC_API_KEY from env

def generate_answer_claude(query: str, chunks: list[dict]) -> str:
    """
    Sends retrieved context chunks and user query to claude and returns an answer

    Args:
        query (str): the user's original question
        chunks (list[dict]): top-k results from retrieve_top_k, each with 'text' and 'meta'

    Returns:
        str: Claude's generated answer
    """
    #puts chunks in one string
    context = "\n\n".join(
        f"[Source: {c['meta']['filename']}, page {c['meta']['page']}]\n{c['text']}"
        for c in chunks
    )

    #query sent to modle
    message = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": (
                f"Answer the question using only the context provided. "
                f"If the answer isn't in the context, say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question: {query}"
            )
        }]
    )

    return message.content[0].text