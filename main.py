import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer =LancasterStemmer()

import numpy
import tflearn
import tensorflow
import random
import json
import pickle

with open("launch.json") as file:
    data=json.load(file)
try:
    with open("data.pickle","rb")as f:
        words,lables,output=pickle.load(f)
except:
    words =[]
    labels =[]
    docs_x =[]
    docs_y=[]

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds =nltk.word_tokenize(pattern)
            print("wrds:")
            print(wrds)
            words.extend(wrds)
            print("words:")
            print(words)
            print("docs_x:")
            docs_x.append(wrds)
            print(docs_x)
            docs_y.append(intent["tag"])
            print(docs_y)
            
        if intent["tag"] not in labels:
            labels.append(intent["tag"])
            print("labels:")
            print(labels)
            print("\n\n\n")
    print("words")
    words=[stemmer.stem(w.lower()) for w in words if w not in ["?","!","."]]
    print(words)
    words=sorted(list(set(words)))
    print(words)
    labels=sorted(labels)

    training=[]
    output=[]

    out_empty=[0 for _ in range(len(labels))]
    print(out_empty)

    for x,doc in enumerate(docs_x):
        bag=[]
        print("doc")
        print(doc)
        wrds=[stemmer.stem(w) for w in doc]
        print("wrds:")
        print(wrds)
        for w in words:
            print(w)
            if w in wrds:
                bag.append(1)
            else:
                bag.append(0)
            print("bag")
            print(bag)
        print("output row:")
        output_row=list(out_empty)
        print(output_row)
        output_row[labels.index(docs_y[x])]=1
        print(output_row)

        training.append(bag)
        output.append(output_row)

        print("train")
        print(training)
        print("output")
        print(output)


    training =numpy.array(training)
    output=numpy.array(output)

    with open("data.pickle","wb")as f:
        pickle.dump((words,labels,training,output),f)
    print(training)
    print(output)
tensorflow.reset_default_graph()

net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,4)
net=tflearn.fully_connected(net,4)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)
try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=4, show_metric=True)
    model.save("model.tflearn")

def bag_of_words(s,words):
    bag=[0 for i in range(len(words))]

    s_words=nltk.word_tokenize(s)
    s_words=[stemmer.stem(word.lower()) for word in s_words]
    for se in s_words:
        for i,w in enumerate(words):
            if w==se:
                bag[i]=1

    return numpy.array(bag)

def chat():
    print("start talking with the bot")
    while True:
        inp=input("You: ")
        if(inp.lower()=="-1"):
            break
        results=model.predict([bag_of_words(inp,words)])[0]
        results_index=numpy.argmax(results)
        tag=labels[results_index]
        if results[results_index] > 0.5:
            for tg in data["intents"]:
                if tg['tag']==tag:
                    response=tg['responses']
            print(random.choice(response))
        else:
            print("I don't understand")
chat()