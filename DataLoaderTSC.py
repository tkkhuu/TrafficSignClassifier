from TKDNNUtil.DataLoader import DataLoader
from sklearn.utils import shuffle

import numpy as np

class DataLoaderTSC(DataLoader):
    
    def GenerateTrainingBatch(self, samples, batch_size=32, flip_images=False, side_cameras=False):
        
        train_samples = samples[0]
        label_samples = samples[1]
        num_samples = len(train_samples)
        while 1: # Loop forever so the generator never terminates
            X, y = shuffle(train_samples, label_samples)
            for offset in range(0, num_samples, batch_size):
                train_batch = train_samples[offset:offset+batch_size]
                label_batch = label_samples[offset:offset+batch_size]
                X_train = np.array(train_batch)
                y_train = np.array(label_batch)
                yield shuffle(X_train, y_train)