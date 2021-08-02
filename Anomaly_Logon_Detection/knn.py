from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt

# loading data - 정상에 비정상 데이터를 concat하여 데이터 셋으로 사용 
df1 = pd.read_csv("user_good.csv")
df2 = pd.read_csv("user_bad.csv")
df = pd.concat([df1,df2])

# feature 선정
feature_cols = ['isnewuser','isnewip','isvpn','islan','percent','src_ip_c']
X = df[feature_cols]
y = df['tag']

# train and split data for testing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.4)
k_range= range(1,26)
scores = []

# Finding the best K 
for k in k_range: 
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    scores.append(metrics.f1_score(y_test, y_pred))
  
    targetname = ['Normal','Malicious']
    result = metrics.classification_report(y_test,y_pred,target_names=targetname)
    matrix = metrics.confusion_matrix(y_test, y_pred, labels = [0,1]) 
    print ('Currently running K:' + str(k))
    print (result)
    print (matrix)

"""
Currently running K:1
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       1.00      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3995    5]
 [   0 1264]]
Currently running K:2
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       1.00      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3995    5]
 [   0 1264]]
Currently running K:3
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3993    7]
 [   0 1264]]
Currently running K:4
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3993    7]
 [   0 1264]]
Currently running K:5
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3988   12]
 [   0 1264]]
Currently running K:6
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      1.00      1264
    accuracy                           1.00      5264
   macro avg       1.00      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3988   12]
 [   0 1264]]
Currently running K:7
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3986   14]
 [   0 1264]]
Currently running K:8
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3986   14]
 [   0 1264]]
Currently running K:9
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3986   14]
 [   0 1264]]
Currently running K:10
              precision    recall  f1-score   support
      Normal       1.00      1.00      1.00      4000
   Malicious       0.99      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      1.00      5264
weighted avg       1.00      1.00      1.00      5264
[[3986   14]
 [   0 1264]]
Currently running K:11
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3978   22]
 [   0 1264]]
Currently running K:12
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3978   22]
 [   0 1264]]
Currently running K:13
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3976   24]
 [   0 1264]]
Currently running K:14
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3976   24]
 [   0 1264]]
Currently running K:15
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3975   25]
 [   0 1264]]
Currently running K:16
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           1.00      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       1.00      1.00      1.00      5264
[[3975   25]
 [   0 1264]]
Currently running K:17
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3973   27]
 [   0 1264]]
Currently running K:18
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3973   27]
 [   0 1264]]
Currently running K:19
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3969   31]
 [   0 1264]]
Currently running K:20
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3969   31]
 [   0 1264]]
Currently running K:21
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.97      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3967   33]
 [   0 1264]]
Currently running K:22
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.98      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3968   32]
 [   0 1264]]
Currently running K:23
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.97      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3963   37]
 [   0 1264]]
Currently running K:24
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.97      1.00      0.99      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3963   37]
 [   0 1264]]
Currently running K:25
              precision    recall  f1-score   support
      Normal       1.00      0.99      1.00      4000
   Malicious       0.97      1.00      0.98      1264
    accuracy                           0.99      5264
   macro avg       0.99      1.00      0.99      5264
weighted avg       0.99      0.99      0.99      5264
[[3961   39]
 [   0 1264]]
"""

# graph 출력  
%matplotlib inline
plt.plot(k_range, scores)
plt.xlabel('Value of K for KNN')
plt.ylabel('Testing Accuracy')
plt.show()

#가장 예측율이 높은 k를 선정 
k = k_range[scores.index(max(scores))] 
print("The best number of k : " + str(k))

"""
The best number of k : 4
"""

# 학습
from sklearn.metrics import accuracy_score

knn = KNeighborsClassifier(n_neighbors=4)

# data split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
k_range= range(1,26)
scores = []

# knn model 학습 
knn.fit(X_train,y_train.values.ravel())
pred = knn.predict(X_test)
print("Acc : " + str(accuracy_score(y_test.values.ravel(),pred)))

"""
Acc : 0.9988601823708206
"""
