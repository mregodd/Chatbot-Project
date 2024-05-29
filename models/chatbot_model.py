import random
import time
import numpy as np
import os
from tensorflow.keras.models import load_model
from models.data_loader import load_intents 
from models.utils import tokenize_and_lemmatize

intents = load_intents()
model = load_model(os.path.join(os.path.dirname(__file__), 'chatbot_model.keras'))

words = []
classes = []
documents = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize_and_lemmatize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

def bow(sentence):
    sentence_words = tokenize_and_lemmatize(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    p = bow(sentence)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if len(results) == 0:
        return [{"intent": "unknown", "probability": "1.0"}]
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

def chatbot_response(msg):
    start_time = time.time()
    ints = predict_class(msg)
    if ints[0]['intent'] == 'unknown':
        return "Üzgünüm, ne demek istediğini anlayamadım."
    else:
        response = get_response(ints, intents)
    end_time = time.time()
    print(f"Response time: {end_time - start_time} seconds")
    return response
