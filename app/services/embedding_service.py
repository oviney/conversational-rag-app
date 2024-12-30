import openai
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import os  # Import the os module

def generate_embedding(text):
    """
    Generate an embedding for the input text using OpenAI's API.
    Uses the OPENAI_API_KEY environment variable for the API key.
    """
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    if not openai.api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    try:
        # Use the new API call format
        client = openai.OpenAI()
        response = client.embeddings.create(
            input=text,
            model="text-embedding-ada-002"
        )
        return response.data[0].embedding
    except Exception as e:
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")

def cosine_similarity(embedding1, embedding2):
    """
    Compute cosine similarity between two embeddings.
    """
    return sklearn_cosine_similarity([embedding1], [embedding2])[0][0]