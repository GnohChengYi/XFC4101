import miceforest as mf
import pandas as pd
import numpy as np

class MiceImputation:
    # generates multiple imputations and takes the average of them
    def inbetween(self, keyframes, num_imputations=5, random_state=4101):
        data_pd = pd.DataFrame(keyframes, columns = [str(i) for i in range(len(keyframes[0]))])
        kds = mf.ImputationKernel(data_pd, save_all_iterations=False, datasets=num_imputations, random_state=random_state)
        kds.mice()
        imputations = []
        for i in range(num_imputations):
            imputation = kds.complete_data(dataset=i).to_numpy()
            imputations.append(imputation)
        np_imputations = np.array(imputations)
        inbetweened_data = np_imputations.mean(axis=0)
        return inbetweened_data
