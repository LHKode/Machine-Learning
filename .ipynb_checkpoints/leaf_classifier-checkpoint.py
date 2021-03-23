import numpy as np
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Flatten, Dense
from tensorflow.keras.models import  Model
from tensorflow.keras.losses import SparseCategoricalCrossentropy
from tensorflow.keras.metrics import SparseCategoricalAccuracy
from tensorflow.keras.optimizers import Nadam, Adam
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


class GestureClassifier:
    def __init__(self):
        self.model = None

    def build_model(self):
        pass # our model
        
    def save_model(self):
       
        pass  # delete this line and replace yours

    def load_model(self):
        pass # load our model
        

    def train(self, train_generator, valid_generator,**kwargs):
        pass

    def predict(self, x_test):
        """
        :param x_test: a numpy array with dimension (N,100,100,3)
        :return: a numpy array with dimension (N,)
        """
        pass
