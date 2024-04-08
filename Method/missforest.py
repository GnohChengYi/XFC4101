from missforest import MissForest

class MissForestImputation:
    def __init__(self):
        self.imputer = MissForest()
    
    def inbetween(self, keyframes):
        inbetweened_data = self.imputer.fit_transform(keyframes)
        return inbetweened_data

    def __str__(self) -> str:
        return "missForest"