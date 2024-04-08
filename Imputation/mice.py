from sklearn.experimental import enable_iterative_imputer   # explicitly enable iterative imputer to use it
from sklearn.impute import IterativeImputer

### taken from https://www.machinelearningplus.com/machine-learning/mice-imputation/

class MiceImputation:
    def inbetween(self, keyframes):
        imputer = IterativeImputer()
        imputer.fit(keyframes)
        inbetweened_data = imputer.transform(keyframes)
        return inbetweened_data
