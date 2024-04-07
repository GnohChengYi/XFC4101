from sklearn.impute import KNNImputer

class KNNImputation:
    def __init__(self, n_neighbors=5):
        self.imputer = KNNImputer(n_neighbors=n_neighbors)
    
    def inbetween(self, keyframes):
        inbetweened_data = self.imputer.fit_transform(keyframes)
        return inbetweened_data
