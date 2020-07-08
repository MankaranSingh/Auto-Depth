import os
import numpy as np
import pandas as pd

def DepthNorm(x, maxDepth):
    return maxDepth / x

def check_data():
    
    depth_filenames = os.listdir("./data/depth")
    rgb_filenames = os.listdir("./data/images")

    if depth_filenames == rgb_filenames and len(depth_filenames) != 0:
      return True
    return False

def get_paths(val_size=0.8):
    
    rgb_filenames = os.listdir("./data/images")
    val_cut = int(np.ceil(len(rgb_filenames)*(1-val_size)))
    return pd.DataFrame({'path':rgb_filenames[:-val_cut]}), pd.DataFrame({'path':rgb_filenames[-val_cut:]})

def rgb_to_depth(image):

    array = image.astype(np.float32)
    normalized_depth = np.dot(array[:, :, :3], [65536.0, 256.0, 1.0])
    normalized_depth /= 16777215.0  
    return normalized_depth

def predict(input):

    pred1 = model.predict(np.array([input]))[0]
    pred2 = np.flip(model.predict(np.array([np.flip(input, 1)]))[0], 1)

    output = (pred1 + pred2) / 2.0
    return output



      
