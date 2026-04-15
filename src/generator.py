'''
Author: Armand Meijers
Date: 12/04/2026
Description: Generates answers from retrieved chunks using llama
'''

import ollama

def generate_answer_llama(query: str, chunks: list[dict]) -> str:
    """
    Sends retrieved context chunks and user query to llama and returns an answer

    Args:
        query (str): the user's original question
        chunks (list[dict]): top-k results from retrieve_top_k, each with 'text' and 'meta'

    Returns:
        str: llama3 generated answer
    """

    response = ollama.chat(
        model='llama3.2:3b',
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant. prioritise context data and use it as your source, but if context is lack luster or not handling main question use internal knowledge aswell but take parts from context'},
            {'role': 'user', 'content': f'Chunk Context: {chunks}\n\n User Question: {query}'}
        ]
    )

    return response['message']['content']
