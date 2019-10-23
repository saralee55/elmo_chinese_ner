# -*- coding: utf-8 -*-
"""
Created on Wed Dec  5 11:05:32 2018

@author: 74284
"""
import jieba
import numpy as np
from elmoformanylangs import Embedder
elmo_model=Embedder()
def get_elmo(text):
    '''
    :param text: list of string
    :return:numpy array of elmo_result,if text_size=1,shape=[1,3,text_length,1024],else shape=[text_size,3]
            numpy array of char_seg,if text_size=1,shape=[1,text_length],else shape=[text_size,]
    '''
    e= elmo_model
    all_text=[]
    for sen in text:
        new_sen=jieba.lcut(sen)
        if len(new_sen)>400:
            new_sen=new_sen[:400]
        all_text.append(new_sen)
    result=e.sents2elmo(all_text,output_layer=1)
    elmo_list=[item.tolist() for item in result]
    elmo_result=np.array(elmo_list,dtype=object)
    char_seg=np.array(all_text,dtype=object)
    return elmo_result,char_seg
   
def get_elmo_batch(text,directory,batch_size):
    '''
    :param text:list of string
    :param directory:path of save_file
    :param batch_size:size of batch,integer
    '''
    e=elmo_model
    all_text=[]
    for sen in text:
        new_sen=jieba.lcut(sen)
        if len(new_sen)>400:
            new_sen=new_sen[:400]
        all_text.append(new_sen)
    length=len(all_text)
    endnum=length-length%batch_size
    j=0
    for i in range(0,length,batch_size):
        j=j+1
        if i+batch_size<=endnum:
            char_text=all_text[i:i+batch_size]
            result=e.sents2elmo(char_text,output_layer=1)
        else:
            char_text=all_text[i:length]
            result=e.sents2elmo(char_text,output_layer=1)
        elmo_filename=directory+'/elmo_result'+str(j)+'.npy'
        char_filename=directory+'/char_text'+str(j)+'.npy'
        elmo_list=[item.tolist() for item in result]
        elmo_result=np.array(elmo_list,dtype=object)
        char_result=np.array(char_text,dtype=object)
        np.save(elmo_filename,elmo_result)
        np.save(char_filename,char_result)
        print('在处理'+str(i)+'篇elmo') 

if __name__ == '__main__':
    #elmo_model=Embedder()
    import datetime
    file=open('newstest.txt','r',encoding='utf-8')
    lines=file.readlines()
    text1=[line for line in lines]
    start=datetime.datetime.now()
    elmo_result,char_seg=get_elmo(text1)
    end=datetime.datetime.now()
    print(end-start)
    #print(elmo_result.shape)
    #print(char_seg)
    #text=[line for line in lines][0:100]
    #get_elmo_batch(text,elmo_model,'test_result',32)
    
    
