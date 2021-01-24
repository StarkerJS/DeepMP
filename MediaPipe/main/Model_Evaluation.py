#!/usr/bin/env python
# coding: utf-8

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
from keras.layers import SimpleRNN, Dense, Dropout
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
import pydot
import graphviz
plt.rcParams["font.family"] = "TakaoGothic"
plt.rc('figure',figsize=[8.0, 4.8])


# In[ ]:


#TRAIN OPTION
framesize  = 19320  #23-*84 18900
inputshape = 230   #126(180) #70(100) 35(50)
epoch      = 300
batchsize  = 40
model_note = 'best_model'
save_path  = 'SaveData/'+ model_note +'/'
os.makedirs(save_path,exist_ok=True)

#TEST OPTION
model_path = 'best_model.h5'
testdatadir= '../../DATASET/Train/Zero/0.6/AbsoluteTEST/'
classname  = ['おはよう','どういたしまして','好き','嫌い','おめでとう','新しい','こんばんは','久しぶり','ありがとう','元気','自由','ウサギ','負け']


# In[ ]:


def load_testdata(dirname):
    listfile=os.listdir(dirname)
    XT = []
    YT = []
    for file in listfile:
        if "_" in file:
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
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
    #make_label(text)
    
    s = Tokenizer()
    s.fit_on_texts([text])
    encoded1=s.texts_to_sequences([YT])[0]
    one_hot2=to_categorical(encoded1)
    
    (x_test,y_test)=XT,one_hot2
    x_test=np.array(x_test)
    y_test=np.array(y_test)
    return x_test,y_test


# In[ ]:


dirname = testdatadir
x_test,y_test=load_testdata(dirname)
new_model = tf.keras.models.load_model(model_path)
new_model.summary()

print('Test stage')
score = new_model.evaluate(x_test,y_test,batch_size=batchsize,verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
print('_________________________________________________________________')
print('Shape & history')
print(x_test.shape)
print(y_test.shape)

xhat = x_test
yhat = new_model.predict(xhat)


# In[ ]:


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.PuBu,linecolor="white", linewidths=10):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size=40)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90, fontsize = 30)
    plt.yticks(tick_marks, classes, fontsize = 30)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black", fontsize = 25)

    plt.tight_layout()
    plt.ylabel('True label', fontsize = 25)
    plt.xlabel('Predicted label', fontsize = 25)
    plt.savefig(save_path+'matrix.png', dpi=300)


# In[ ]:


cfm = confusion_matrix(np.argmax(y_test,axis=1), np.argmax(yhat, axis=1))
np.set_printoptions(precision=2)

plt.figure(figsize=(30,20))
class_names = classname
plot_confusion_matrix(cfm, classes=class_names, title='')
plt.show()

