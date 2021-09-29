# ipynb로 분석을 진행하였으나, 편의를 위해 py로 업로드 

import pandas as pd
import tensorflow as tf

"""
from google.colab import drive
drive.mount('/content/drive')

cd drive/MyDrive/TEST/
"""

df = pd.read_feather('data/logs.ft')

# df.columns
# df.shape

fields = ['UtcTime', 'ProcessId', 'EventID', 'User', 'Image', 'ImageLoaded', 'CommandLine',  'ParentImage', 'ParentCommandLine', 'DestinationPort', 'Protocol', 'QueryName', 'TargetFilename', 'TargetObject', 'raw']
newdf = df[fields]

# newdf

# drop all records where ProcessId in NaN (happens for WMI events, cannot classify [TODO: think how to overcome and add to dataset])
newdf = newdf[~newdf.ProcessId.isna()]

# drop EventID 5 - ProcessTerminated as not valuable
newdf.drop(newdf[newdf.EventID == '5'].index, inplace=True)


# Feature Engineering

## Image Processing

# get binary name (last part of "Image" after "\")
newdf['binary'] = newdf.Image.str.split(r'\\').apply(lambda x: x[-1].lower())

# same with binary pathes
newdf['path'] = newdf.Image.str.split(r'\\').apply(lambda x: '\\'.join(x[:-1]).lower())

"""
print('Total different unique binaries:', newdf['binary'].nunique())
newdf['binary'][0:10]

print('Total different unique paths:', newdf['path'].nunique())
newdf['path'][0:10]
"""

## CommandLine Arguments
newdf['arguments'] = newdf.CommandLine.fillna('empty').str.split().apply(lambda x: ' '.join(x[1:]))
newdf['arguments'][-5:]



## contains base64

# add new features whether suspicious string are in arguments?
# 1. base64?
import re

# will match at least 32 character long consequent string with base64 characters only
b64_regex = r"[a-zA-Z0-9+\/]{64,}={0,2}"

# test on some
newdf['arguments'][newdf['arguments'].str.contains('enc')]

# there's matches
for i in newdf['arguments'][newdf['arguments'].apply(lambda x: re.search(b64_regex, x)).notnull()]:
    print(i,"\n")
  
# map this search as 0 and 1 using astype(int)
b64s = newdf['arguments'].apply(lambda x: re.search(b64_regex, x)).notnull()
newdf['b64'] = b64s.astype(int)
b64s[b64s]


## URL & UNC paths?
# matches if there's call for some file with extension (at the end dot) via UNC path
unc_regex = r"\\\\[a-zA-Z0-9]+\\[a-zA-Z0-9\\]+\."
uncs = newdf['arguments'][newdf['arguments'].apply(lambda x: re.search(unc_regex, x)).notnull()]
# we didn't had any of these launches in dataset btw
uncs[uncs].index

url_regex = r"https?:\/\/(www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b([-a-zA-Z0-9()@:%_\+.~#?&//=]*)"
urls = newdf['arguments'].apply(lambda x: re.search(url_regex, x)).notnull()
urls[urls]

# verified pd.concat part - merges two boolean series correctly
newdf['unc_url'] = pd.concat([uncs, urls]).astype(int)

# check if correct marking
newdf[newdf['unc_url'].astype(bool)]


## Network connection
newdf['network'] = newdf['Protocol'].notnull().astype(int)


# preprocessed data save
import pandas as pd
newdf.to_csv("newdf.csv")


# data load
newdf = pd.read_csv("newdf.csv")


# Model 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.sequence import pad_sequences
from datetime import datetime
import numpy as np

newdf = newdf[['ProcessId','binary','path', 'unc_url', 'b64', 'network']]
# treat eventID as int8
#newdf['EventID'] = newdf['EventID'].astype('int8')
#newdf.head()

newdf["binary"].value_counts()
"""
powershell.exe                 654
svchost.exe                    550
backgroundtaskhost.exe         318
runtimebroker.exe              229
microsoftedge.exe              189
wmiprvse.exe                   146
browser_broker.exe             128
calculator.exe                 114
localbridge.exe                107
conhost.exe                    103
dllhost.exe                     96
tiworker.exe                    91
notepad.exe                     90
whoami.exe                      78
fodhelper.exe                   71
wscript.exe                     70
microsoftedgecp.exe             70
taskhostw.exe                   62
katz.exe                        62
microsoftedgesh.exe             58
consent.exe                     55
trustedinstaller.exe            52
wmiapsrv.exe                    40
systeminfo.exe                  39
fsatps.exe                      38
searchui.exe                    37
sysmon.exe                      27
system                          20
schtasks.exe                    18
applicationframehost.exe        16
netstat.exe                     16
explorer.exe                    15
fsatpn.exe                      14
wsqmcons.exe                    13
smartscreen.exe                 11
fshoster32.exe                   6
mmc.exe                          3
services.exe                     2
searchfilterhost.exe             1
startmenuexperiencehost.exe      1
fshoster64.exe                   1
fsorsp64.exe                     1
Name: binary, dtype: int64
"""

# LabelEncoder
le = LabelEncoder()
le = le.fit(newdf["binary"])
newdf["binary"] = le.transform(newdf["binary"])

# binary 매핑 결과 출력 
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

newdf["path"] = newdf["path"].astype(str)
le = le.fit(newdf["path"])
newdf["path"] = le.transform(newdf["path"])

#path 매핑한 결과 출력
le_name_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(le_name_mapping)

# 라벨인코딩한 버전 저장 
import pandas as pd
data.to_csv("data2.csv")

# tag 추가한 데이터 load
data = pd.read_csv("data3.csv")

del data["Unnamed: 0"]
del data["ProcessId"]

# Model
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
df = data
# feature used 
feature_cols = ["EventID","binary",	"path",	"b64",	"unc_url",	"network"]
X = df[feature_cols]
y = df['tag']
# train and split data for testing 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)

le = LabelEncoder()
le.fit(y_train)
print(le.classes_) #['N' 'Y']
#labels = le.classes_

targetname = ['Normal','Malicious']

model = KNeighborsClassifier(n_neighbors=2)
cv_results = cross_validate(model, X_train, y_train, return_estimator=True) #내부적으로 fit

x_test = [
    ['3','22','10','0','0','1']
]

x_test = pd.DataFrame(x_test, columns=["EventID","binary","path","b64","unc_url","network"])

best_index = np.argsort(cv_results['test_score'])[-1]
best_model = cv_results['estimator'][best_index]

joblib.dump(best_model, 'Procmodel.pkl')

import joblib
import os
import pandas as pd

MODEL_PATH = "Procmodel.pkl"

class ProcModel:

    def __init__(self):
        self.model = self._load_model_from_path(MODEL_PATH)

    @staticmethod
    def _load_model_from_path(path):
        model = joblib.load(path)
        return model

    def predict(self, data, return_option='Prob'):
        #df = pd.DataFrame(data)
        df = pd.DataFrame(x_test, columns=["EventID","binary","path","b64","unc_url","network"])
        if return_option == 'Prob':
            predictions = self.model.predict(df) #추후 추가
        else:
            predictions = self.model.predict(df)
        return predictions
      
model = ProcModel()
x_test = [
    ['3','32','15','0','0','1']
]
targetname = ['Normal','Malicious']
y_predict = model.predict(x_test)
label = targetname[y_predict[0]]
#model.predict(x_test)
print(label)



