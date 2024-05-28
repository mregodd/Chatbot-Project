# -*- coding: utf-8 -*-
import random
import json
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
import numpy as np
import os

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

try:
    current_directory = os.path.dirname(os.path.abspath(__file__))
    intents_file_path = os.path.join(current_directory, '..', 'data', 'intents.json')
    with open(intents_file_path, encoding='utf-8') as file:
        intents = json.load(file)
except json.JSONDecodeError as e:
    print(f"JSONDecodeError: {e}")
    exit()
except FileNotFoundError as e:
    print(f"FileNotFoundError: {e}")
    exit()

model = load_model(os.path.join(current_directory, 'chatbot_model.keras'))

words = []
classes = []
documents = []
ignore_words = ['?', '!', '.', ',']

synonyms = {
    "sağ ol": "sağol",
    "teşekkürler": "teşekkür"
}

def normalize_sentence(sentence):
    for word, norm in synonyms.items():
        sentence = sentence.replace(word, norm)
    return sentence

for intent in intents['intents']:
    for pattern in intent['patterns']:
        pattern = normalize_sentence(pattern)
        word_list = nltk.word_tokenize(pattern)
        words.extend(word_list)
        documents.append((word_list, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

def clean_up_sentence(sentence):
    sentence = normalize_sentence(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.75
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    
    if len(results) == 0:
        return [{"intent": "unknown", "probability": "1.0"}]
    
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def get_response(ints, intents_json):
    tag = ints[0]['intent']
    if tag == 'unknown':
        return random.choice([i['responses'][0] for i in intents_json['intents'] if i['tag'] == 'unknown'])
    else:
        list_of_intents = intents_json['intents']
        for i in list_of_intents:
            if i['tag'] == tag:
                result = random.choice(i['responses'])
                break
        return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = get_response(ints, intents)
    return res
