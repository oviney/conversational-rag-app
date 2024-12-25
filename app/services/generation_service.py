import logging
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GenerationService:
    def __init__(self, model_name="gpt2"):
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
            logging.info(f"Model {model_name} loaded successfully.")
        except Exception as e:
            logging.error(f"Initialization error: {str(e)}")
            raise RuntimeError(f"Failed to initialize generation model: {str(e)}")
        
    def generate_text(self, context: str, prompt: str, max_new_tokens: int = 150, temperature: float = 0.7) -> str:
        if not context and not prompt:
            raise ValueError("Both context and prompt are empty.")
        
        try:
            combined_input = f"Context:\n{context}\n\nPrompt:\n{prompt}\n\nResponse:"
            input_ids = self.tokenizer.encode(combined_input, return_tensors="pt").to(self.model.device)
            
            # Truncate input to model's max length
            max_input_length = self.tokenizer.model_max_length - max_new_tokens
            input_ids = input_ids[:, -max_input_length:]
            
            # Generate response
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9,  # Enable nucleus sampling
                top_k=50,   # Limit to top 50 token candidates
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            response = response.replace(context, "").strip()  # Remove context from response
            response = re.sub(r"[^\x00-\x7F]+", "", response)  # Sanitize output
            
            logging.debug(f"Generated response: {response}")
            if not response.strip():
                return "I'm sorry, I couldn't generate a response."
            
            return response
        except Exception as e:
            logging.error(f"Generation error: {str(e)}")
            raise ValueError(f"Failed to generate response: {str(e)}")