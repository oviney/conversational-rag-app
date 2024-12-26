import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GenerationService:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, context: str, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        if not context and not prompt:
            raise ValueError("Both context and prompt are empty.")
        
        try:
            combined_input = f"Context:\n{context}\n\nInstruction:\n{prompt}\n\nResponse:"
            logging.debug(f"Combined Input: {combined_input}")
            
            input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.model.device)
            logging.debug(f"Input IDs: {input_ids}")
            
            # Truncate input to model's max length
            max_input_length = self.tokenizer.model_max_length - max_new_tokens
            input_ids = input_ids[:, -max_input_length:]
            logging.debug(f"Truncated Input IDs: {input_ids}")
            
            # Generate response
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.95,  # Adjust top_p for better quality
                top_k=40,    # Adjust top_k for better quality
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=1.2,  # Add repetition penalty
                eos_token_id=self.tokenizer.eos_token_id  # Ensure generation stops at EOS token
            )
            logging.debug(f"Output: {output}")
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True).strip()  # Remove leading/trailing whitespace
            logging.debug(f"Decoded Response: {response}")
            response = response.replace(combined_input, "").strip()  # Remove combined input from response
            response = re.sub(r"[^\x00-\x7F]+", "", response)  # Sanitize output
            
            logging.debug(f"Sanitized Response: {response}")
            if not response:
                return "I'm sorry, I couldn't generate a response."
            
            return response
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")

def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.to("cpu")  # Explicitly move the model to CPU
    return model, tokenizer