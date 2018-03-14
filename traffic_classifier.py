import numpy as np
from DataLoader import DataLoader
from keras.utils import to_categorical
from sklearn.utils import shuffle

import DNNBuilder

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

data_loader = DataLoaderTSC()

# Preparing data
train_file = '../traffic-signs-data/train.p'
valid_file = '../traffic-signs-data/valid.p'
test_file = '../traffic-signs-data/test.p'

train = data_loader.LoadPickleData(train_file)
valid = data_loader.LoadPickleData(valid_file)
test = data_loader.LoadPickleData(test_file)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']

print("Succesfully imported data\n")

# Store data properties into variables
n_train = X_train.shape[0]
n_validation = X_valid.shape[0]
n_test = X_test.shape[0]
image_shape = X_train[0].shape
n_classes = 43

BATCH_SIZE = 16
EPOCHS = 8



print('Train data shape: {}'.format(X_train.shape))
print('Train label shape: {}'.format(y_train.shape))
print('Number of training examples: {}'.format(n_train))
print('Number of validation examples: {}'.format(n_validation))
print('Number of testing examples: {}'.format(n_test))
print('Image data shape: {}'.format(image_shape))
print('Number of classes: {}\n'.format(n_classes))

one_hot_y_train = to_categorical(y_train, num_classes=n_classes)
one_hot_y_valid = to_categorical(y_valid, num_classes=n_classes)

train_generator = data_loader.GenerateTrainingBatch((X_train, one_hot_y_train))
valid_generator = data_loader.GenerateTrainingBatch((X_valid, one_hot_y_valid))

def normalize(data):
    return (data - 128.0) / 128.0

traffic_dnn_layers = (
                        {'layer_type': 'lambda', 'function': normalize},
                        {'layer_type': 'convolution', 'shape': (5, 5, 16), 'padding': 'valid', 'activation': 'relu'},
                        {'layer_type': 'convolution', 'shape': (5, 5, 32), 'padding': 'valid', 'activation': 'relu'},
                        {'layer_type': 'dropout', 'keep_prob': 0.4},
                        {'layer_type': 'flatten'},
                        {'layer_type': 'dropout', 'keep_prob': 0.4},
                        {'layer_type': 'fully connected', 'output_dim': 400, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'output_dim': 120, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'output_dim': 84, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'output_dim': n_classes, 'activation': 'softmax'}
                     )

print('Creating Network ...')
traffic_dnn = DNNBuilder.DNNSequentialModelBuilder2D({'input_shape': X_train.shape[1:], 'model_architecture': traffic_dnn_layers})

print('Initializing Network ...')
traffic_dnn.Initialize()

print('Compiling Network ...')
traffic_dnn.Compile('categorical_crossentropy', 'adam')



print('Training Network ...')
traffic_dnn.FitAndSave(train_generator, valid_generator, n_train, n_validation, EPOCHS)

print('Evaluate Model ...')
traffic_dnn.Evaluate(X_valid, y_valid)


