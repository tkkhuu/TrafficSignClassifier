from keras.models import Sequential
from keras.layers.convolutional import Convolution2D
from keras.layers import Dense, Activation, Dropout, Flatten, Lambda, Cropping2D

class DNNSequentialModelBuilder:
    def __init__(self, architecture):
        self._input_shape = architecture['input_shape']
        self._model_architecture = architecture['model_architecture']
        self._model_type = 'Sequential'
        self._model = None
        
    def _AddConv(self, detail, first_layer):
        raise NotImplementedError('Convolution layer has not been implemented')
    
    def _AddCropping(self, detail, first_layer):
        raise NotImplementedError('Cropping layer has not been implemented')
    
    # Core Layers
    def _AddFullyConnected(self, detail, first_layer):
        output_dim = detail['output_dim']
        
        try: init = detail['weight_init_func']
        except KeyError: init = 'glorot_uniform'
            
        try: activation = detail['activation']
        except KeyError: activation = 'linear'
        
        try: weights = detail['init_weights']
        except KeyError: weights = None
        
        try: W_regularizer = detail['weight_regularizer']
        except KeyError: W_regularizer = None
        
        try: b_regularizer = detail['bias_regularizer']
        except KeyError: b_regularizer = None
        
        try: activity_regularizer = detail['activity_regularizer']
        except KeyError: activity_regularizer = None
        
        try: W_constraint = detail['weight_constraint']
        except: W_constraint = None
        
        try: b_constraint = detail['bias_constraint']
        except KeyError: b_constraint = None
        
        try: input_dim = detail['input_dim']
        except: input_dim = None
        
        if first_layer:
            fc = Dense(output_dim,
                       init = init,
                       activation = activation,
                       weights = weights,
                       W_regularizer = W_regularizer,
                       b_regularizer = b_regularizer,
                       activity_regularizer = activity_regularizer,
                       W_constraint = W_constraint,
                       b_constraint = b_constraint,
                       input_dim = input_dim,
                       input_shape = self._input_shape)
        else:
            fc = Dense(output_dim,
                       init = init,
                       activation = activation,
                       weights = weights,
                       W_regularizer = W_regularizer,
                       b_regularizer = b_regularizer,
                       activity_regularizer = activity_regularizer,
                       W_constraint = W_constraint,
                       b_constraint = b_constraint,
                       input_dim = input_dim)
        
        self._model.add(fc)
        print('Added fully connected layer')
    
    def _AddActivation(self, detail, first_layer):
        activation = detail['activation']
        
        if first_layer:
            self._model.add(Activation(activation, input_shape=self._input_shape))
        else:
            self._model.add(Activation(activation))
        
        print('Added activation function: {}'.format(activation))
    
    def _AddDropout(self, detail, first_layer):
        
        keep_prob = detail['keep_prob']
        
        if first_layer:
            raise ValueError("Dropout can't be the first layer")
        
        self._model.add(Dropout(keep_prob))
        print('Added dropout with keep probability {}'.format(keep_prob))
    
    def _AddFlatten(self, detail, first_layer):
        
        if first_layer:
            flatten = Flatten(input_shape=self._input_shape)
        else:
            flatten = Flatten()
            
        self._model.add(flatten)
        print('Added flatten layer')
    
    def _AddLambda(self, detail, first_layer):
        func = detail['function']
        if first_layer:
            self._model.add(Lambda(func, input_shape=self._input_shape))
        else:
            self._model.add(Lambda(func))
            
        print('Added lambda layer with function: {}'.format(func.__name__))
    
    def _AddLayer(self, detail, first_layer=False):
         
        if detail['layer_type'] == 'convolution':
            self._AddConv(detail, first_layer)
        
        elif detail['layer_type'] == 'flatten':
            self._AddFlatten(detail, first_layer)
        
        elif detail['layer_type'] == 'fully connected':
            self._AddFullyConnected(detail, first_layer)
        
        elif detail['layer_type'] == 'lambda':
            self._AddLambda(detail, first_layer)
        
        elif detail['layer_type'] == 'cropping':
            self._AddCropping(detail, first_layer)
            
        elif detail['layer_type'] == 'dropout':
            self._AddDropout(detail, first_layer)
        
        else:
            raise NotImplementedError('This layer is not supported: {}'.format(detail['layer_type']))   
    
    def Initialize(self):
        print('Adding Layers ....................')
        self._model = Sequential()
        for idx, dic in enumerate(self._model_architecture):
            if idx == 0: self._AddLayer(dic, first_layer=True)
            else: self._AddLayer(dic)
        print('Done Adding Layers ....................\n')
    
    def Compile(self, loss, optimizer, metrics=None, sample_weight_mode=None, weighted_metrics=None, target_tensors=None):
        self._model.compile(loss=loss, optimizer=optimizer, metrics=metrics, sample_weight_mode=sample_weight_mode, weighted_metrics=weighted_metrics, target_tensors=target_tensors)
        
    def Train(self, X_train, y_train, validation=0.2, shuffle=True):
        self._model.fit(X_train, y_train, validation_split=validation, shuffle=shuffle)
        
    def FitAndSave(self, train_generator, valid_generator, n_train, n_valid, n_epoch):
        self._model.fit_generator(train_generator,
                                   samples_per_epoch=n_train,
                                   validation_data=valid_generator,
                                   nb_val_samples=n_valid,
                                   nb_epoch=n_epoch)
        
        self._model.save('../model.h5')
        print('************ Saved as model.h5 ************')
        
    def TrainAndSave(self, x_train, y_label, class_weight=None, sample_weight=None):
        self._model.train_on_batch(x_train, y_label, class_weight=None, sample_weight=None)
        #self._model.save('../model')
        
    def Evaluate(self, x=None, y=None, batch_size=None, verbose=1, sample_weight=None, steps=None):
        self._model.evaluate(x, y, batch_size=None, verbose=1, sample_weight=None, steps=None)
    
    def __str__(self):
        print('============ DNN Architecture ============')
        result = '''Input Shape: {}\nModel Type: {}\nArchitecture:\n'''. format(self._input_shape, self._model_type)
        
        for layer in self._model_architecture:
            layer_str = '{}:  '.format(layer['layer_type'])
            for k, v in layer.items():
                
                if k != 'layer_type':
                    layer_str += '{}={} '.format(k, v)
            layer_str += '\n'
            result += layer_str
        return result

class DNNSequentialModelBuilder2D(DNNSequentialModelBuilder):
    
    def __init__(self, architecture):
        super().__init__(architecture)
        
    def _AddConv(self, detail, first_layer):
        
        # Required parameters
        nb_filter = detail['shape'][2]
        nb_row = detail['shape'][0]
        nb_col = detail['shape'][1]
        
        # Optional parameters
        try: init = detail['weight_init_func']
        except KeyError: init = 'glorot_uniform'
            
        try: activation = detail['activation']
        except KeyError: activation = 'linear'
        
        try: weights = detail['init_weights']
        except KeyError: weights = None
        
        try: border_mode = detail['padding']
        except KeyError: border_mode = 'valid'
        
        try: subsample = detail['stride']
        except KeyError: subsample = (1, 1)
        
        try: dim_ordering = detail['dim_ordering']
        except KeyError: dim_ordering = 'tf'
        
        try: W_regularizer = detail['weight_regularizer']
        except KeyError: W_regularizer = None
        
        try: b_regularizer = detail['bias_regularizer']
        except KeyError: b_regularizer = None
        
        try: activity_regularizer = detail['activity_regularizer']
        except KeyError: activity_regularizer = None
        
        try: W_constraint = detail['weight_constraint']
        except: W_constraint = None
        
        try: b_constraint = detail['bias_constraint']
        except KeyError: b_constraint = None
        
        if first_layer:
            conv = Convolution2D(nb_filter,
                                 nb_row,
                                 nb_col,
                                 init = init,
                                 activation = activation,
                                 weights = weights,
                                 border_mode = border_mode,
                                 subsample = subsample,
                                 dim_ordering = dim_ordering,
                                 W_regularizer = W_regularizer,
                                 b_regularizer = b_regularizer,
                                 activity_regularizer = activity_regularizer,
                                 W_constraint = W_constraint,
                                 b_constraint = b_constraint,
                                 input_shape=self._input_shape)
        
        else:
            conv = Convolution2D(nb_filter,
                                 nb_row,
                                 nb_col,
                                 init = init,
                                 activation = activation,
                                 weights = weights,
                                 border_mode = border_mode,
                                 subsample = subsample,
                                 dim_ordering = dim_ordering,
                                 W_regularizer = W_regularizer,
                                 b_regularizer = b_regularizer,
                                 activity_regularizer = activity_regularizer,
                                 W_constraint = W_constraint,
                                 b_constraint = b_constraint)
        
        self._model.add(conv)
        print('Added convolution layer')
    
    def _AddCropping(self, detail, first_layer):
        cropping = detail['crop_dim']
        
        try: dim_ordering = detail['dim_ordering']
        except KeyError: dim_ordering = 'tf'
        
        if first_layer:
            self._model.add(Cropping2D(cropping=cropping, dim_ordering=dim_ordering, input_shape=self._input_shape))
        else:
            self._model.add(Cropping2D(cropping=cropping, dim_ordering=dim_ordering))
        
        print('Added cropping layer')
        