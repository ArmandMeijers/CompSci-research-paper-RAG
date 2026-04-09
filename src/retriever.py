'''
Author: Armand Meijers
Date: 02/04/2026
Description: Prompts user for query and embeddeds it 
'''

from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

# Load model globally once
model = SentenceTransformer("BAAI/bge-large-en-v1.5")

def prompt_user_query() -> tuple[str, np.ndarray]:
    """
    Prompts the user for a question and returns its embedding vector.

    Returns:
        np.ndarray: Query embedding with shape (1, embedding_dim), dtype=float32
    """
    
    #prompt user
    query = input("What is your query?: ").strip()
    if not query:
        print("[WARN] Empty query entered.")
        return "", np.zeros((1, model.get_sentence_embedding_dimension()), dtype="float32")
    
    embeddings = model.encode([query])
    user_vector = np.array(embeddings, dtype="float32").reshape(1, -1)
    return query, user_vector

def cosine_similarity(index, query_vector, top_k=5):
    """
    Search a faiss index using cosine similarity.

    Args:
        index: faiss index (IndexFlatL2)
        query_vector: np.ndarray of shape (1, dim)
        top_k: number of results to return

    Returns:
        distances, indices: np.ndarray
    """
    
    # FAISS expects float32 arrays
    query_vector = np.array(query_vector, dtype="float32")
    
    # normalize if using cosine similarity with IndexFlatL2
    faiss.normalize_L2(query_vector)
    
    distances, indices = index.search(query_vector, top_k)
    return distances, indices