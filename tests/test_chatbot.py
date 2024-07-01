import pytest
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from models.chatbot_model import bow, predict_class, get_response, chatbot_response
from models.data_loader import load_intents
from models.utils import tokenize_and_lemmatize

intents = load_intents()

def test_tokenize_and_lemmatize():
    sentence = "Merhaba, nasılsın?"
    tokens = tokenize_and_lemmatize(sentence)
    assert tokens == ['merhaba', 'nasıl', 'ol']

def test_bow():
    sentence = "Merhaba"
    bag = bow(sentence)
    assert len(bag) == len(set([word for intent in intents['intents'] for pattern in intent['patterns'] for word in tokenize_and_lemmatize(pattern)]))

def test_predict_class():
    sentence = "Merhaba"
    classes = predict_class(sentence)
    assert isinstance(classes, list)
    assert 'intent' in classes[0]
    assert 'probability' in classes[0]

def test_get_response():
    ints = [{"intent": "selamlama", "probability": "1.0"}]
    response = get_response(ints, intents)
    assert response in ["Merhaba!", "Selam!", "Selamlar, hoşgeldiniz."]

def test_chatbot_response():
    from models.chatbot_model import chatbot_response
    response = chatbot_response("Merhaba")
    assert response in ["Merhaba!", "Selam!", "Selamlar, hoşgeldiniz."]

if __name__ == "__main__":
    pytest.main()
