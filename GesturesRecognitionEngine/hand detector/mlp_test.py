#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 18:26:17 2019

@author: ajithkumar
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#from sklearn.model_selection import train_test_split

heart = pd.read_csv('centers.csv')
X_tr=[]
Y_tr=[]
X = heart.iloc[0:9, 0:2]
Y = heart.iloc[0:9, 2]
Y=pd.Series.tolist(Y)
for i in range(9):
    X_base=pd.Series.tolist(X.iloc[i,:])
    xl_lim=int(X_base[0])-25
    xu_lim=int(X_base[0])+25
    yl_lim=int(X_base[1])-25
    yu_lim=int(X_base[1])+25
    for xlim in range(xl_lim,xu_lim,1):
        for ylim in range(yl_lim,yu_lim,1):
            X_tr.append([xlim,ylim])
            Y_tr.append(int(Y[i]))
            
X_test=heart.iloc[9:19, 0:2]
Y_test=heart.iloc[9:19, 2]
    
    

#print(X)

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, Y_test = train_test_split(X_tr, Y_tr, test_size=0, random_state=0)

#print(X_train.shape)
#print(X_test.shape)

#print(Y_train.shape)
#print(Y_test.shape)

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
sc.fit(X_tr)
X_train_std = sc.transform(X_tr)

X_test_std = sc.transform(X_test)

from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=100, tol=1e-4, eta0=0.0001, fit_intercept=True, random_state=0, verbose=True)
ppn.fit(X_train_std, Y_tr)
#ppn.fit(X_tr,Y_tr)

#y_pred = ppn.predict(X_test_std)
y_pred = ppn.predict(X_test)

print('Misclassified samples: %d' %(Y_test != y_pred).sum())

from sklearn.metrics import accuracy_score
print('Accuracy: %0.3f' % accuracy_score(Y_test, y_pred))
