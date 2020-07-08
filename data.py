import os
import numpy as np
import pandas as pd
from skimage.transform import resize
from keras.utils.data_utils import Sequence
from utils import DepthNorm
from augmentation import augment, flip
from utils import rgb_to_depth
import cv2

class DataGenerator2D(Sequence):
    """Generates data for Keras
    Sequence based data generator. Suitable for building data generator for training and prediction.
    """
    def __init__(self, img_paths, base_path, to_fit=True, batch_size=32, shuffle=True, debug_dir='', debug_imgs=[], augmentation_rate=0.5,
                 max_depth=100, min_depth=0):
    
        self.img_paths = img_paths.copy()
        self.base_path = base_path
        self.to_fit = to_fit
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.debug_dir = debug_dir
        self.debug_imgs = debug_imgs
        self.epoch = 0
        self.augmentation_rate = augmentation_rate
        self.max_depth = max_depth            # Meter 
        self.min_depth = min_depth            # Meter 
        self.on_epoch_end()
    
    def __len__(self):
        """Denotes the number of batches per epoch
        :return: number of batches per epoch
        """
        return int(len(self.img_paths) // self.batch_size)

    def __getitem__(self, index):
        """Generate one batch of data
        :param index: index of the batch
        :return: X and y when fitting. X only when predicting
        """
        # Generate indexes of the batch
        current_indexes = list(range(index * self.batch_size, (index + 1) * self.batch_size))
        img_paths_temp = self.img_paths[current_indexes]

        # Generate data
        X = []
        y = []

        for path in img_paths_temp:
          _X = cv2.cvtColor(cv2.imread(self.base_path + f"/images/{path}"), cv2.COLOR_BGR2RGB)
          _y = rgb_to_depth(cv2.imread(self.base_path + f"/depth/{path}"))
          _y = 1000.0*_y

          if (np.random.random() < self.augmentation_rate):
            _X = augment(_X)

          if (np.random.random() < 0.5) and self.augmentation_rate:
            _X, _y = flip(_X, _y)
          
          _y = np.clip(_y, self.min_depth, self.max_depth) 
          _y = DepthNorm(_y, maxDepth=self.max_depth) 

          _y = resize(_y, (_X.shape[0]//2, _X.shape[1]//2), preserve_range=True, mode='reflect', anti_aliasing=True )
          _y = _y.reshape(_y.shape[0], _y.shape[1], 1)
          #_y = np.log(_y)

          X.append(_X)
          y.append(_y)
            
        if self.to_fit:
            return (np.array(X)/255).astype('float32'), np.array(y).astype('float32')
        else:
            return np.array(X).astype('float32')

    def on_epoch_end(self):
        """Updates indexes after each epoch
        """
        if self.shuffle == True:
            indices = np.arange(len(self.img_paths))
            np.random.shuffle(indices)
            self.img_paths = self.img_paths[indices]
            self.img_paths.reset_index(drop=True, inplace=True)

        if self.debug_dir:
          preds = model.predict(np.array(self.debug_imgs))
          cmap = plt.get_cmap('plasma')

          for i, pred in enumerate(preds):
            pred = pred.reshape(pred.shape[0], pred.shape[1])
            location = os.path.join(self.debug_dir, f"{self.epoch}-{i}.jpg")
            cv2.imwrite(location, pred*255)

          self.epoch += 1


      
