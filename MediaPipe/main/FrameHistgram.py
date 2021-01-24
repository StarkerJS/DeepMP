#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import xlsxwriter
import os
import sys
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pylab as plt
plt.rc('figure',figsize=[8.0, 4.8])
plt.rcParams["font.family"] = "TakaoGothic"


# In[ ]:


def rm_zero_sequences_endlist(txtlist):
    numbers = txtlist
    if (numbers[-1] ==0 and numbers[-2] ==0 and numbers[-3] ==0 and numbers[-4] ==0 and numbers[-5] ==0 and numbers[-6] ==0
    and numbers[-7] ==0 and numbers[-8] ==0 and numbers[-9] ==0 and numbers[-10] ==0 and numbers[-11] ==0 and numbers[-12] ==0
    and numbers[-13] ==0 and numbers[-14] ==0 and numbers[-15] ==0 and numbers[-16] ==0 and numbers[-17] ==0 and numbers[-18] ==0
    and numbers[-19] ==0 and numbers[-20] ==0 and numbers[-21] ==0 and numbers[-22] ==0 and numbers[-23] ==0 and numbers[-24] ==0
    and numbers[-25] ==0 and numbers[-26] ==0 and numbers[-27] ==0 and numbers[-28] ==0 and numbers[-29] ==0 and numbers[-30] ==0
    and numbers[-31] ==0 and numbers[-32] ==0 and numbers[-33] ==0 and numbers[-34] ==0 and numbers[-35] ==0 and numbers[-36] ==0
    and numbers[-37] ==0 and numbers[-38] ==0 and numbers[-39] ==0 and numbers[-40] ==0 and numbers[-41] ==0 and numbers[-42] ==0):
        numbers = numbers[:-42]
    return numbers


# In[ ]:


def return_frame(dirname):
    frames=[] #list to save frame numbers in txt files
    listfile=os.listdir(dirname)
    for file in listfile:
        if "_" in file: #ignore mp4 files
            continue
        wordname=file
        textlist=os.listdir(dirname+wordname)
        for text in textlist:
            if "DS_" in text:
                continue
            textname=dirname+wordname+"/"+text
            with open(textname, mode = 'r') as t: #open txt files 
                numbers = np.array([float(num) for num in t.read().split()])
                while numbers[0] == 0:
                    numbers = numbers[1:]
                #print(int(len(numbers)/84))
                frames.append(int(len(numbers)/84))
                

    count = Counter(frames)
    print(count)
    plt.bar(count.keys(), count.values())
    plt.ylim(0, 130)
    plt.xlim(0, 250)
    plt.xticks(np.arange(0, 230 + 1, 46))
    plt.xlabel('フレーム数')
    plt.ylabel('データ数')
    plt.title('フレーム数分布ヒストグラム（片手手話パディング前）')
    plt.grid(True)
    #plt.savefig('defslow-hist.png', dpi=300)


# In[ ]:


return_frame('../../DATASET/Data/defslow/Absolute/')


# In[ ]:




