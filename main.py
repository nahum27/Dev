import nltk
from nltk.stem.lancaster import LancasterStemmer

#nltk.download('all')
#LookupError 발생시 nltk.download()로 필요한 데이터를 매칭 시켜 해결

stemmer = LancasterStemmer()

import numpy
#import tflearn
#import tensorflow
import random
import json


with open("./json/intents.json") as file:
    data = json.load(file)



words = []
labels = []
docs = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
        docs.append(pattern)

    if intent["tag"] not in labels:
        labels.append(intent['tag'])



print(words)
print(labels)
print(docs)