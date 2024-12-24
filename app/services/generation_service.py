# Handles text generation using Hugging Face models
from transformers import pipeline


class GenerationService:
    def __init__(self):
        self.generator = pipeline(
            'text-generation',
            model=Config.GENERATION_MODEL)

    def generate_text(self, context, query, max_length=200):
        prompt = f"Context: {context}\nQuery: {query}\nResponse:"
        output = self.generator(
            prompt,
            max_length=max_length,
            num_return_sequences=1)
        return output[0]['generated_text'].split("Response:")[1].strip()
