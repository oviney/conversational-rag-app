import pytest
from unittest.mock import MagicMock
from app.services.generation_service import GenerationService
import torch

@pytest.fixture
def mock_model_and_tokenizer():
    mock_tokenizer = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer.encode.return_value = torch.tensor([[1, 2, 3]])
    mock_tokenizer.decode.return_value = "Generated response"
    mock_tokenizer.model_max_length = 512
    mock_tokenizer.eos_token_id = 50256
    mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5]])
    mock_model.generate = MagicMock()
    mock_model.device = torch.device("cpu")
    return mock_model, mock_tokenizer

@pytest.fixture
def generation_service(mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer
    return GenerationService(model, tokenizer)

def test_generate_text_success(generation_service):
    context = "This is a context."
    prompt = "This is a prompt."
    response = generation_service.generate_text(context, prompt)
    assert response == "Generated response"

def test_generate_text_empty_context_and_prompt(generation_service):
    with pytest.raises(ValueError, match="Both context and prompt are empty."):
        generation_service.generate_text("", "")

def test_generate_text_model_generate_failure(generation_service, mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer
    model.generate.side_effect = Exception("Model error")
    with pytest.raises(ValueError, match="Failed to generate response: Model error"):
        generation_service.generate_text("context", "prompt")

def test_generate_text_empty_response(generation_service, mock_model_and_tokenizer):
    model, tokenizer = mock_model_and_tokenizer
    tokenizer.decode.return_value = ""
    response = generation_service.generate_text("context", "prompt")
    assert response == "I'm sorry, I couldn't generate a response."