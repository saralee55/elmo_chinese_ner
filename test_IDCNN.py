# -*- coding: utf-8 -*-
"""
Created on Thu Dec  6 11:07:29 2018

@author: 74284
"""

import pandas as pd
from evaluate_model import test_evaluate
#file=pd.read_excel('train_test_5000_new3.xlsx',encoding='utf-8')
#text=file['文本'][0:1]
file=open('newstest.txt','r',encoding='utf-8')
lines=file.readlines()
text=[line for line in lines][0:1]
test_evaluate(text)
text2=[line for line in lines]
test_evaluate(text2)
#print(t1)
#text=file['文本'][0:32]
#test_evaluate(text)
#print(t2)
#text=file['文本'][0:100]
#test_evaluate(text)
#print(t2)


