from app.config import Config
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging


class GenerationService:
    def __init__(self):
        self.model_name = "gpt2"  # or your preferred model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
        
        # Configure tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"
        
    def generate_text(self, context: str, query: str) -> str:
        if not context and not query:
            raise ValueError("Both context and query are empty.")
        try:
            # Prepare input
            prompt = f"Context: {context}\nQuestion: {query}\nAnswer:"
            
            # Tokenize with proper settings
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            )
            
            # Generate
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                temperature=0.7
            )
            
            # Decode and format response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract only the answer part
            if "Answer:" in response:
                response = response.split("Answer:")[-1].strip()
            
            logging.debug(f"Generated response: {response}")
            return response
            
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")
