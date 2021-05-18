# 해당 코드는 jupyter lab에서 작성되었으며, 편의상 Py로 업드로함.


import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import json
import gc
import printr
import csv
import os
import sys
import glob
import shutil
import pprint
import pytz
import logging
import re
import urllib.request
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

pd.set_option('display.max_row', 500)
pd.set_option('display.max_columns', 100)

data = pd.read_csv("data.csv") # 데이터의 경우 pkl파일을 편의상 Csv파일로 변환한 다음 사용

# 악성의 cmdline 확인
data[data['type'] == 1]['cmdline'].value_counts()

# 정상의 cmdline 확인
data[data['type'] == 0]['cmdline'].value_counts()

data['type'] = data['type'].astype(int) 
data['type'].value_counts().plot(kind='bar');
print(data.groupby('type').size().reset_index(name='count'))

X_data = data['cmdline']
y_data = data['type']

print('cmdline: {}'.format(len(X_data)))
print('type : {}'.format(len(y_data)))


tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # 21090개의 행을 가진 X의 각 행에 토큰화를 수행
sequences = tokenizer.texts_to_sequences(X_data) # 단어를 숫자값, 인덱스로 변환하여 저장

#print(sequences[:5])

word_to_index = tokenizer.word_index
#print(word_to_index)

threshold = 3
total_cnt = len(word_to_index) # 단어의 수
rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

# 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
for key, value in tokenizer.word_counts.items():
    total_freq = total_freq + value

    # 단어의 등장 빈도수가 threshold보다 작으면
    if(value < threshold):
        rare_cnt = rare_cnt + 1
        rare_freq = rare_freq + value

print('frequency less than rare_cnt: %s'%(threshold - 1, rare_cnt))
print('rare word rate in vocabulary :', (rare_cnt / total_cnt)*100)
print("Rate of frequency of rare word appearance in total frequency of appearance:", (rare_freq / total_freq)*100)

vocab_size = len(word_to_index) + 1
print('vocab size: {}'.format((vocab_size)))


n_of_train = int(len(sequences) * 0.8)
n_of_test = int(len(sequences) - n_of_train)
print('train data :',n_of_train)
print('test data:',n_of_test)

X_data = sequences
print('cmdline max len : %d' % max(len(l) for l in X_data))
print('cmdline average len : %f' % (sum(map(len, X_data))/len(X_data)))
plt.hist([len(s) for s in X_data], bins=50)
plt.xlabel('length of samples')
plt.ylabel('number of samples')
plt.show()

max_len = 80 # 전체 데이터셋의 길이는 max_len으로 맞춤
data = pad_sequences(X_data, maxlen = max_len)
print("train data size(shape): ", data.shape)

X_test = data[n_of_train:] #X_data 데이터 중에서 뒤의 n개의 데이터만 저장
y_test = np.array(y_data[n_of_train:]) #y_data 데이터 중에서 뒤의 n개의 데이터만 저장
X_train = data[:n_of_train] #X_data 데이터 중에서 앞의 n개의 데이터만 저장
y_train = np.array(y_data[:n_of_train]) #y_data 데이터 중에서 앞의 n개의 데이터만 저장

from tensorflow.keras.layers import SimpleRNN, Embedding, Dense
from tensorflow.keras.models import Sequential

model = Sequential()
model.add(Embedding(vocab_size, 16)) # 임베딩 벡터의 차원은 32
model.add(SimpleRNN(16)) # RNN 셀의 hidden_size는 32
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=3, batch_size=16, validation_split=0.2)

print("\n 테스트 정확도: %.4f" % (model.evaluate(X_test, y_test)[1]))

epochs = range(1, len(history.history['acc']) + 1)
plt.plot(epochs, history.history['loss'])
plt.plot(epochs, history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

