import TKDNNUtil.DNNBuilder as DNNBuilder
from DataLoaderTSC import DataLoaderTSC
from keras.utils import to_categorical

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
                        {'layer_type': 'fully connected', 'units': 400, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'units': 120, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'units': 84, 'activation': 'relu'},
                        {'layer_type': 'fully connected', 'units': n_classes, 'activation': 'softmax'}
                     )

print('Creating Network ...')
traffic_dnn = DNNBuilder.DNNSequentialModelBuilder2D({'input_shape': X_train.shape[1:], 'model_architecture': traffic_dnn_layers})

print('Initializing Network ...')
traffic_dnn.Initialize()

print('Compiling Network ...')
traffic_dnn.Compile('categorical_crossentropy', 'adam')

print('Training Network ...')
traffic_dnn.FitGenerator(train_generator, validation_data=valid_generator, steps_per_epoch=n_train, validation_steps=n_validation, epochs=EPOCHS)

print('Evaluate Model ...')
traffic_dnn.Evaluate(X_valid, y_valid)


