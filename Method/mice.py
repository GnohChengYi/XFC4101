from sklearn.experimental import enable_iterative_imputer   # explicitly enable iterative imputer to use it
from sklearn.impute import IterativeImputer

class MiceImputation:
    # generates multiple imputations and takes the average of them
    def inbetween(self, keyframes, n_imputations=5):
        imputer = IterativeImputer()
        imputer.fit(keyframes)
        imputations = []
        for _ in range(n_imputations):
            imputed_data = imputer.transform(keyframes)
            imputations.append(imputed_data)
        
        inbetweened_data = sum(imputations) / n_imputations
        return inbetweened_data

    def __str__(self) -> str:
        return "MICE"
