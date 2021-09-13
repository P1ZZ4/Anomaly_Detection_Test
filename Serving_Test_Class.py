import joblib
import os
import pandas as pd

MODEL_PATH = "best_model.pkl"

class SysModel:

    """ Wrapper for loading and serving pre-trained model"""

    def __init__(self):
        self.model = self._load_model_from_path(MODEL_PATH)

    @staticmethod
    def _load_model_from_path(path):
        model = joblib.load(path)
        return model

    def predict(self, data, return_option='Prob'):
        #df = pd.DataFrame(data)
        df = pd.DataFrame(x_test, columns=['islan','isnewip','isnewuser','isvpn','percent','src_ip_c','total_c'])
        if return_option == 'Prob':
            predictions = self.model.predict(df) #prob error # 추후 구현
        else:
            predictions = self.model.predict(df)
        return predictions
      


model = SysModel()

x_test = [
    ['0','0','0','0','0.9999882','121212','121212']
] 

#model.predict(x_test)
targetname = ['Normal','Malicious']
y_predict = model.predict(x_test)
label = targetname[y_predict[0]]
print(label)

#Normal
