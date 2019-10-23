# -*- coding: utf-8 -*-
"""
Created on Thu Nov 22 11:37:45 2018

@author: 74284
"""

import numpy as np

def convert_feature_tag(sentence,char_to_id,feature,tag):
    elmo=[]
    for ed in feature:
        line=[]
        for i in range(len(ed[0])):
            linesmall=[]
            linesmall.append(ed[0][i])
            linesmall.append(ed[1][i])
            linesmall.append(ed[2][i])
            line.append(linesmall)
        elmo.append(line)
    batch_size=len(elmo)
    #print(batch_size)
    batch=[]        
    label=tag
    sentences=sentence.tolist()
    maxlength=max([len(item) for item in sentences])
    chars=[]
    for sen in sentences:
        char=[]
        for word in sen:
            if word in char_to_id:
                char.append(char_to_id[word])
            else:
                char.append(char_to_id['<UNK>'])
        chars.append(char)
    charpad=[]
    strpad=[]
    labelpad=[]
    elmopad=[]
    for n in range(len(sentences)):
        if maxlength-len(sentences[n])==0:
            strpad.append(sentences[n])
            charpad.append(chars[n])
            labelpad.append(label[n])
            elmopad.append(elmo[n])
        else:
            padding=[0] * (maxlength - len(label[n]))
            charpad.append(chars[n]+padding)
            strpad.append(sentences[n]+padding)
            labelpad.append(label[n]+padding)
            elmopadding=[[[0]*1024]*3] * (maxlength- len(label[n]))
            elmopad.append(elmo[n]+elmopadding)
    layer1=[]
    layer2=[]
    layer3=[]
    for line in elmopad:
        word=[]
        word1=[]
        word2=[]
        for item in line:
            word.append(item[0])
            word1.append(item[1])
            word2.append(item[2])
        layer1.append(word)
        layer2.append(word1)
        layer3.append(word2)
    np_elmopad=np.array(elmopad)
    print(np_elmopad.shape)
    #reshape_elmopad=np.array(elmopad).reshape(batch_size,maxlength,3072)
    #featurepad=reshape_elmopad.tolist()
    batch.append(strpad)
    batch.append(charpad)
    batch.append(layer1)
    batch.append(layer2)
    batch.append(layer3)
    batch.append(labelpad)
    
    return batch
