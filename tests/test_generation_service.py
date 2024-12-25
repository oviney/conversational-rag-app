import pytest
from unittest.mock import Mock, patch
from app.services.generation_service import GenerationService

@pytest.fixture
def generation_service():
    with patch('transformers.AutoTokenizer.from_pretrained') as mock_tokenizer, \
         patch('transformers.AutoModelForCausalLM.from_pretrained') as mock_model:
        
        mock_tokenizer.return_value = Mock()
        mock_tokenizer.return_value.eos_token = "[EOS]"
        mock_tokenizer.return_value.pad_token = None
        
        mock_model.return_value = Mock()
        mock_model.return_value.generate.return_value = [[1, 2, 3]]
        
        service = GenerationService()
        service.tokenizer.decode = Mock(return_value="Test response")
        return service

def test_generate_text_success(generation_service):
    context = "Test context"
    query = "Test query"
    response = generation_service.generate_text(context, query)
    assert isinstance(response, str)
    assert response == "Test response"

def test_generate_text_with_empty_input(generation_service):
    with pytest.raises(ValueError):
        generation_service.generate_text("", "")

def test_generate_text_model_error(generation_service):
    generation_service.model.generate.side_effect = Exception("Model error")
    with pytest.raises(ValueError, match="Failed to generate response"):
        generation_service.generate_text("context", "query")