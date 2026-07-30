import joblib
import pandas as pd

class PredictiveMaintenance:

    def __init__(self,model_path):

        self.model=joblib.load(model_path)

    def predict(self,data):

        probability=self.model.predict_proba(data)[0][1]

        prediction=self.model.predict(data)[0]

        return prediction,probability
