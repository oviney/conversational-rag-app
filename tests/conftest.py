import os
import spacy
import pytest


@pytest.fixture(scope='session', autouse=True)
def load_spacy_model():
    # Load the Spacy model
    try:
        nlp = spacy.load('en_core_web_sm')
        print("Spacy model 'en_core_web_sm' loaded successfully.")
    except Exception as e:
        raise RuntimeError(f"Failed to load Spacy model: {e}")

    return nlp
