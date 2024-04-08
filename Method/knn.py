from sklearn.impute import KNNImputer

class KNNImputation:
    def __init__(self):
        self.imputer = KNNImputer()
    
    def inbetween(self, keyframes):
        inbetweened_data = self.imputer.fit_transform(keyframes)
        return inbetweened_data
