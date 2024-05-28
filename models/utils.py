import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
nltk.download('punkt')
nltk.download('wordnet')

synonyms = {
    "sağ ol": "sağol",
    "teşekkürler": "teşekkür"
}

ignore_words = ['?', '!', '.', ',']

def normalize_sentence(sentence):
    for word, norm in synonyms.items():
        sentence = sentence.replace(word, norm)
    return sentence

def tokenize_and_lemmatize(sentence):
    sentence = normalize_sentence(sentence)
    sentence_words = nltk.word_tokenize(sentence)
    return [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
