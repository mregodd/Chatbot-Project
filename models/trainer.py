import os
import sys
import json
import numpy as np
import random
import nltk
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.optimizers import SGD
import matplotlib.pyplot as plt


current_directory = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_directory, '..'))

from models.data_loader import load_intents
from models.utils import tokenize_and_lemmatize, ignore_words, lemmatizer

nltk.download('punkt')
nltk.download('wordnet')

def train_model():
    intents = load_intents()
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

    training = []
    output_empty = [0] * len(classes)

    for doc in documents:
        bag = []
        pattern_words = doc[0]
        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]
        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1
        training.append([bag, output_row])

    random.shuffle(training)
    training = np.array(training, dtype=object)

    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    print(f"Number of words (features): {len(words)}") 
    print(f"Classes: {classes}") 

    model = Sequential()
    model.add(Input(shape=(len(train_x[0]),)))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(learning_rate=0.01, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)
    
    model.save(os.path.join(current_directory, '..', 'chatbot_model.keras')) 

    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['loss'])
    plt.title('Model accuracy & loss')
    plt.ylabel('Accuracy / Loss')
    plt.xlabel('Epoch')
    plt.legend(['Accuracy', 'Loss'], loc='upper left')
    plt.show()

    print("Model oluşturuldu ve kaydedildi")

if __name__ == "__main__":
    train_model()
