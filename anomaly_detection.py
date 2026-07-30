from sklearn.ensemble import IsolationForest

class AnomalyDetector:

    def __init__(self):

        self.model=IsolationForest(

            contamination=0.02,

            random_state=42

        )

    def train(self,X):

        self.model.fit(X)

    def detect(self,X):

        return self.model.predict(X)
