import random
import time
import numpy as np
import os
import json
import logging
from tensorflow.keras.models import load_model
import sys


current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

from models.data_loader import load_intents 
from models.utils import tokenize_and_lemmatize

logging.basicConfig(level=logging.ERROR, format='%(asctime)s %(levelname)s %(message)s') 
intents = load_intents()
model = load_model(os.path.join(current_directory,  'chatbot_model.keras')) 

words = []
classes = []

for intent in intents['intents']:
    for pattern in intent['patterns']:
        word_list = tokenize_and_lemmatize(pattern)
        words.extend(word_list)
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = sorted(list(set(words)))
classes = sorted(list(set(classes)))

print(f"Number of words (features): {len(words)}")  
print(f"Classes: {classes}")

def bow(sentence):
    sentence_words = tokenize_and_lemmatize(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    try:
        p = bow(sentence)
        res = model.predict(np.array([p]))[0]
        ERROR_THRESHOLD = 0.75
        results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
        results.sort(key=lambda x: x[1], reverse=True)
        
        if len(results) == 0:
            return [{"intent": "unknown", "probability": "1.0"}]
        
        return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]
    except Exception as e:
        logging.error(f"Error in predict_class: {e}")
        return [{"intent": "unknown", "probability": "1.0"}]

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    for i in intents_json['intents']:
        if i['tag'] == tag:
            return random.choice(i['responses'])
    return None

def chatbot_response(msg):
    try:
        start_time = time.time()
        ints = predict_class(msg)
        if ints[0]['intent'] == 'unknown':
            return "Üzgünüm, ne demek istediğini anlayamadım."
        else:
            response = get_response(ints, intents)
        end_time = time.time()
        logging.info(f"Response time: {end_time - start_time} seconds") 
        return response
    except Exception as e:
        logging.error(f"Error in chatbot_response: {e}")
        return "Üzgünüm, bir hata oluştu."
