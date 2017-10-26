#!/usr/bin/env python
# -*- coding: utf-8 -*-

# from __future__ import print_function

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import Conv1D, GlobalMaxPooling1D
from keras import backend as K

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM

from utils import TextLoader

utils_dir = 'utils'
data_path = 'data/data_2w_arti_clean.csv'
test_data_path = 'data/data_test_arti_1_clean.csv'

# set parameters:
max_features = 5000
maxlen = 500
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 60

def f1score(y_true, y_pred):
    num_tp = K.sum(y_true*y_pred)
    num_fn = K.sum(y_true*(1.0-y_pred))
    num_fp = K.sum((1.0-y_true)*y_pred)
    num_tn = K.sum((1.0-y_true)*(1.0-y_pred))
    #print num_tp, num_fn, num_fp, num_tn
    f1 = 2.0*num_tp/(2.0*num_tp+num_fn+num_fp)
    return f1

print('Loading data...')

data_loader = TextLoader(True, utils_dir, data_path, batch_size, 20, None, None)
data_test_loader = TextLoader(True, utils_dir, test_data_path, batch_size, 20, None, None)

x_train = data_loader.tensor_xa
y_train = data_loader.tensor_y

x_test = data_test_loader.tensor_xa
y_test = data_test_loader.tensor_y

# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
print(len(x_train), 'train sequences')
print(len(x_test), 'test sequences')


print('Pad sequences (samples x time)')
x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

print('Build model...')
embedding_vecor_length = 32
model = Sequential()
model.add(Embedding(max_features, embedding_vecor_length, input_length=maxlen))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=[f1score])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
validation_data=(x_test, y_test))