import nltk
from nltk.stem.lancaster import LancasterStemmer

#nltk.download('all')
#LookupError 발생시 nltk.download()로 필요한 데이터를 매칭 시켜 해결

stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
#tensorflow 2.0버전은 tensorflow.contrib이 없다 pip installl tensorflow=1.14버전
#ModuleNotFoundError: No module named 'tensorflow.contrib'
import random
import json
import pickle

with open("./json/intents.json") as file:
    data = json.load(file)


try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)

except:
    words = []
    labels = []
    docs = []
    docs_x = []
    docs_y = []


    for intent in data['intents']:
        for pattern in intent['patterns']:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent['tag'])
        if intent["tag"] not in labels:
            labels.append(intent['tag'])



    words = [stemmer.stem(w.lower()) for w in words if w !="?" ]
    words = sorted(list(set(words)))

    labels = sorted(labels)


    training = []
    output = []

    out_empty = [0 for _ in range(len(labels))]


    for x, doc in enumerate(docs_x):
        bag = []

        wrds = [stemmer.stem(w) for w in doc]

        for w in words:
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)

            output_row = out_empty[:]
            output_row[labels.index(docs_y[x])] = 1

        training.append(bag)
        output.append(output_row)

    training = numpy.array(training)
    output = numpy.array(output)

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.reset_defaault_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net,8)
net = tflearn.fully_connected(net, len(output[0]), activation='softmax')

model = tflearn.DNN(net)


try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save('model.tflearn')











print(words)
print(labels)
print(docs_x)
print(docs_y)