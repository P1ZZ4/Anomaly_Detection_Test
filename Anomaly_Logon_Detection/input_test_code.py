from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
%matplotlib inline

# data concat 
df1 = pd.read_csv("user_log_1.csv")
df2 = pd.read_csv("user_log_2.csv")
df = pd.concat([df1,df2])

# 사용할 col 선정
feature_cols = ['isnewuser','isnewip','isvpn','islan','percent','src_ip_c','total_c']

# x, y 분류
x = df[feature_cols]
y = df['tag']

# test, train split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2)

# data 확인
print(x_train.head())
print(x_train.columns) 

"""
      isnewuser  isnewip  isvpn  islan   percent  src_ip_c  total_c
1848          1        1      0      0  1.000000         1        1
4834          0        0      0      0  0.999553    120745   120799
8595          0        0      0      0  0.998735     58425    58499
9661          0        0      0      0  0.999553    120745   120799
1401          0        0      0      0  0.999553    120745   120799
Index(['isnewuser', 'isnewip', 'isvpn', 'islan', 'percent', 'src_ip_c',
       'total_c'],
      dtype='object')
"""

print(y_train.head())

"""
1848    1
4834    0
8595    0
9661    0
1401    0
"""

# y값 label
le = LabelEncoder()
le.fit(y_train)
print(le.classes_)

targetname = ['Normal','Malicious']

y_train = le.transform(y_train)
y_test = le.transform(y_test)


# knn model 사용
model = KNeighborsClassifier(n_neighbors=5)

cv_results = cross_validate(model, x_train, y_train, return_estimator=True) #내부적으로 fit

print(cv_results['test_score'].mean()) 
df = pd.DataFrame(cv_results)
df = df.sort_values(by='test_score', ascending=False)
print(df)

"""
0.9932559162487904
   fit_time  score_time               estimator  test_score
0  0.019300    0.092420  KNeighborsClassifier()    0.994302
3  0.014217    0.076463  KNeighborsClassifier()    0.994299
2  0.016750    0.090022  KNeighborsClassifier()    0.993827
1  0.015330    0.081953  KNeighborsClassifier()    0.993352
4  0.014516    0.077341  KNeighborsClassifier()    0.990499
"""

## 임의의 값 테스트 (tag = Normal)
x_test = [
    ['0','0','0','0','0.999552976','120745','120799']
]
x_test = pd.DataFrame(x_test, columns=['islan','isnewip','isnewuser','isvpn','percent','src_ip_c','total_c'])

best_index = np.argsort(cv_results['test_score'])[-1]
best_model = cv_results['estimator'][best_index]
y_predict = best_model.predict(x_test)
label = targetname[y_predict[0]]
y_predict = best_model.predict_proba(x_train)
confidence = y_predict[0][y_predict[0].argmax()]
print(label, confidence) # Normal 0.96

