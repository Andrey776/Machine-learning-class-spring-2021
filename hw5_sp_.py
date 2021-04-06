#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  5 16:08:28 2021

@author: AM
"""
import numpy as np 
from sklearn.decomposition import PCA
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.preprocessing import StandardScaler


%cd /Users/AM/Documents/_CU Masters/2021 spr Python ML_5027/ML_py_code/hw_data

f = sio.loadmat('faces.mat')
X = f['X']

# plt.figure(figsize=(10, 10), dpi=100)

# for i in range(100):
#     digit_image = X[i,:].reshape((32,32), order = 'F')
#     plt.subplot(10,10,i+1)
#     plt.imshow(digit_image, cmap = mpl.cm.gist_gray)
    
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
    
scaler = StandardScaler()
Xs = scaler.fit_transform(X)
pca = PCA()
pca.fit(X)

pca1 = PCA(n_components=300)
pca1.fit(Xs)

Xs_reduced = pca1.transform(Xs)
Xs_recovered = pca1.inverse_transform(Xs_reduced)

plt.figure(figsize=(12, 12), dpi=100)

for i in range(100):
    digit_image = Xs[i,:].reshape((32,32), order = 'F')
    plt.subplot(10,10,i+1)
    plt.imshow(digit_image, cmap = mpl.cm.gist_gray)
    
plt.figure(figsize=(12, 12), dpi=100)

for i in range(100):
    digit_image = Xs_recovered[i,:].reshape((32,32), order = 'F')
    plt.subplot(10,10,i+1)
    plt.imshow(digit_image, cmap = mpl.cm.gist_gray)



