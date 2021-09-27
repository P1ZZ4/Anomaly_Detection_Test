from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
#from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_validate
import joblib
%matplotlib inline

# loading data - for my case, I divided my dataset into good data and malicious data 
df1 = pd.read_csv("user_good.csv")
df2 = pd.read_csv("user_bad.csv")
df = pd.concat([df1,df2])
# feature used 
feature_cols = ['isnewuser','isnewip','isvpn','islan','percent','src_ip_c']
X = df[feature_cols]
y = df['tag']
# train and split data for testing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

le = LabelEncoder()
le.fit(y_train)
print(le.classes_) #['N' 'Y']
#labels = le.classes_
targetname = ['Normal','Malicious']
model = KNeighborsClassifier(n_neighbors=5)
cv_results = cross_validate(model, x_train, y_train, return_estimator=True) #내부적으로 fit

x_test = [
    ['0','0','0','0','0.999552976','120745','120799']
]


x_test = pd.DataFrame(x_test, columns=['islan','isnewip','isnewuser','isvpn','percent','src_ip_c'])

best_index = np.argsort(cv_results['test_score'])[-1]

best_model = cv_results['estimator'][best_index]

#--------------------------------------------------------------------------

joblib.dump(best_model, 'best_model2.pkl')

#--------------------------------------------------------------------------

# Testing

import joblib
import os
import pandas as pd

MODEL_PATH = "best_model.pkl"

class SysModel:

    def __init__(self):
        self.model = self._load_model_from_path(MODEL_PATH)

    @staticmethod
    def _load_model_from_path(path):
        model = joblib.load(path)
        return model

    def predict(self, data, return_option='Prob'):
        #df = pd.DataFrame(data)
        df = pd.DataFrame(x_test, columns=['islan','isnewip','isnewuser','isvpn','percent','src_ip_c'])
        if return_option == 'Prob':
            predictions = self.model.predict(df) #추후 추가
        else:
            predictions = self.model.predict(df)
        return predictions
      
model = SysModel()
x_test = [
    ['0','0','0','0','0.9999882','121212']
] 

targetname = ['Normal','Malicious']
y_predict = model.predict(x_test)
label = targetname[y_predict[0]]
#model.predict(x_test)
print(label)

