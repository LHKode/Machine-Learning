import numpy as np
from tensorflow.keras.utils import Sequence
from imgaug import augmenters as iaa
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

class DataGenerator(Sequence):
    """
    Data Generator for Image
    """
    def __init__(self, batch_size, np_images, np_labels):
        self.batch_size = batch_size
        self.np_labels = np_labels
        self.np_images = np_images

        self.num_images = len(self.np_images)
        
        self.preprocesser = iaa.Sequential([
            
            
            
            
            ],
            random_order= True
        )

        self.indices = np.random.permutation(self.num_images)

    def on_epoch_end(self):
        self.indices = np.random.permutation(self.num_images)

    def __len__(self):
        return int(self.num_images / self.batch_size)

    def __getitem__(self, index):
        batch_indices = self.indices[index * self.batch_size: (index+1) * self.batch_size]
        batch_images = self.np_images[batch_indices]
        batch_labels = self.np_labels[batch_indices]

        batch_preprocessed_images = self.preprocesser(images = batch_images)

        
        data_images = preprocess_input(np.concatenate([batch_preprocessed_images, batch_images]))
        data_labels = np.concatenate([batch_labels, batch_labels])
        return data_images, data_labels
