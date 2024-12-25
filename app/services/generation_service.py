from app.config import Config
# Handles text generation using Hugging Face models
from transformers import pipeline
import logging


class GenerationService:
    def __init__(self):
        self.generator = pipeline(
            'text-generation',
            model=Config.GENERATION_MODEL,
            device=0  # Use the first GPU
        )

    def generate_text(self, context, query, max_length=200):
        logging.debug(f"Generating text with context: {context} and query: {query}")
        prompt = f"Context: {context}\nQuery: {query}\nResponse:"
        output = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1)
        response = output[0]['generated_text'].split("Response:")[1].strip()
        logging.debug(f"Generated text: {response}")
        return response
