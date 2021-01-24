#!/usr/bin/env python
# coding: utf-8

# ### hyperopt - hyperas

# In[ ]:


from __future__ import absolute_import, division, print_function, unicode_literals
from keras.preprocessing import sequence
from keras.datasets import imdb
from keras import layers, models
from keras.models import Sequential
from keras import layers
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
import random
from keras import optimizers
from keras.layers import SimpleRNN, Dense, Dropout, Activation
from keras.layers import Bidirectional
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard, EarlyStopping
import matplotlib.pyplot as plt
import os
import sys
import argparse
import itertools
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score
from sklearn.metrics import classification_report
import seaborn as sns
import pandas as pd
import pprint
from hyperopt import Trials, STATUS_OK, tpe
from hyperas import optim
from hyperas.distributions import choice


# In[ ]:


def prepare_data():
    framesize  = 19320
    inputshape = 230
    dirname='../../DATASET/Train/Zero/0.6/AbsoluteTRAIN/'
    if dirname[-1]!='/':
        dirname=dirname+'/'
    listfile=os.listdir(dirname)
    X = []
    Y = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file 
        textlist=os.listdir(dirname+wordname)
        k=0
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            numbers=[]
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                while numbers[0] == 0:
                    numbers = numbers[1:]
                for i in range(len(numbers),framesize):
                    numbers.extend([0.000]) 
            row=0
            landmark_frame=[]
            for i in range(0,inputshape):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            X.append(np.array(landmark_frame))         
            Y.append(wordname)
    X=np.array(X)
    Y=np.array(Y)
    tmp = [[x,y] for x, y in zip(X, Y)]
    random.shuffle(tmp)
    X = [n[0] for n in tmp]
    Y = [n[1] for n in tmp]
    k=set(Y)
    ks=sorted(k)
    text=""
    for i in ks:
        text=text+i+" "
    s = Tokenizer()
    s.fit_on_texts([text])
    encoded=s.texts_to_sequences([Y])[0]
    one_hot = to_categorical(encoded)
    (x_train, y_train) = X, one_hot
    x_train=np.array(x_train)
    y_train=np.array(y_train)
    
    dirnameval='../../DATASET/Train/Zero/0.6/AbsoluteVAL/'
    listfile=os.listdir(dirnameval)
    XT = []
    YT = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirnameval+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirnameval+wordname+"/"+text
            numbers=[]
            with open(textname, mode = 'r') as t:
                numbers = [float(num) for num in t.read().split()]
                while numbers[0] == 0:
                    numbers = numbers[1:]
                for i in range(len(numbers),framesize):
                    numbers.extend([0.000]) 
            landmark_frame=[]
            row=0
            for i in range(0,inputshape):
                landmark_frame.extend(numbers[row:row+84])
                row += 84
            landmark_frame=np.array(landmark_frame)
            landmark_frame=landmark_frame.reshape(-1,84)
            XT.append(np.array(landmark_frame))
            YT.append(wordname)
    XT=np.array(XT)
    YT=np.array(YT)

    tmp1 = [[xt,yt] for xt, yt in zip(XT, YT)]
    random.shuffle(tmp1)

    XT = [n[0] for n in tmp1]
    YT = [n[1] for n in tmp1]
    
    k=set(YT)
    ks=sorted(k)
    text=""
    for i in ks:
        text=text+i+" "
    
    s = Tokenizer()
    s.fit_on_texts([text])
    encoded1=s.texts_to_sequences([YT])[0]
    one_hot2=to_categorical(encoded1)
    
    (x_test,y_test)=XT,one_hot2
    x_test=np.array(x_test)
    y_test=np.array(y_test)

    return x_train,y_train,x_test,y_test


# In[ ]:


def create_model(x_train, y_train, x_test, y_test):
    epoch      = 300
    batchsize  = 40
    model = Sequential()
    model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True, input_shape=(inputshape, 84)))
    if {{choice(['two', 'three', 'four', 'five'])}} == 'two':
        pass
    elif {{choice(['two', 'three', 'four', 'five'])}} == 'three':
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
    elif {{choice(['two', 'three', 'four', 'five'])}} == 'four':
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
    elif {{choice(['two', 'three', 'four', 'five'])}} == 'five':
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
        model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}, return_sequences=True))
    
    model.add(layers.LSTM({{choice([32, 64, 128, 256, 512])}}))
    model.add(layers.Dense(14,activation='softmax'))
    model.summary()
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['acc'])
    model.fit(x_train, y_train,
              batch_size=batchsize,
              epochs=epoch,
              verbose=1,
              validation_data=(x_test, y_test))
    val_loss, val_acc = model.evaluate(x_test, y_test, verbose=0)
    return {'loss': -val_loss, 'status': STATUS_OK, 'model': model}


# In[ ]:


if __name__ == "__main__":
    print('Training stage')
    best_run, best_model = optim.minimize(model=create_model,
                                          data=prepare_data,
                                          algo=tpe.suggest,
                                          max_evals=100,
                                          notebook_name='Hyperopt',
                                          trials=Trials())
    print(best_model.summary())
    print(best_run)
    best_model.save('best_model.h5')
    _, _, x_test, y_test = prepare_data()
    val_loss, val_acc = best_model.evaluate(x_test,y_test,batch_size=40,verbose=0)
    print("val_loss: ", val_loss)
    print("val_acc: ", val_acc)


# In[ ]:




