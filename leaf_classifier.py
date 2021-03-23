import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.models import  Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class LeafClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        input_layer = Input(shape = (256, 256, 3))

        conv_layer_1 = (Conv2D(16, kernel_size = 2, activation="relu"))(input_layer)

        max_pooling_2d_a = (MaxPooling2D(pool_size=(2, 2)))(conv_layer_1)

        conv_layer_2 = (Conv2D(8, kernel_size = 2, activation="relu"))(max_pooling_2d_a)

        max_pooling_2d_b = (MaxPooling2D(pool_size=(2, 2)))(conv_layer_2)

        flat = Flatten()(max_pooling_2d_b)

        dense_layer_1 = Dense(10, activation ='relu')(flat)

        output_layer = Dense(6, activation = 'softmax')(dense_layer_1)

        self.model = Model(input_layer, output_layer)
        print(self.model.summary())
        
        loss = SparseCategoricalCrossentropy()
        optimizer = Adam(learning_rate = 1e-4)
        
        self.model.compile(loss=loss, optimizer=optimizer, metrics=[SparseCategoricalAccuracy()])
        
    def save_model(self):
       
        pass  # delete this line and replace yours

    def load_model(self):
         self.model = load_model('model/leaf.hdf5')
        

    def train(self, train_generator, valid_generator,**kwargs):
        save_best = ModelCheckpoint(
          'model/leaf.hdf5',monitor = 'val_sparse_categorical_accuracy',save_best_only = True, verbose = 1 )
        self.model.fit_generator(
            train_generator,
            validation_data=valid_generator,
            steps_per_epoch=50,
            workers=2,
            epochs=50,
            validation_steps = 1,
            callbacks = [save_best]
        )

    def predict(self, x_test):
        """
        :param x_test: a numpy array with dimension (N,100,100,3)
        :return: a numpy array with dimension (N,)
        """
        x = preprocess_input(x_test)
        y_predict = self.model.predict(x)
        return y_predict.argmax(axis=1)
