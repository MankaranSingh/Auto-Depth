from skimage.util import random_noise
from PIL import Image, ImageEnhance, ImageOps
from itertools import product, permutations
import numpy as np

def add_noise(image):
    return (random_noise(image/255)*255).astype('int')

def color_change(image):
  indices = list(permutations(range(3), 3))  
  idx = np.random.randint(0, len(indices) - 1)
  return image[..., indices[idx]]

def flip(image, depth):
                                                                     
  image, depth = np.flip(image, 1), np.flip(depth, 1)         # Horizontal
  if np.random.random() < 0.5:                                # Vertical
    image, depth = np.flip(image, 0), np.flip(depth, 0)

  return image, depth

def eraser(input_img, p=0.3, s_l=0.02, s_h=0.4, r_1=0.3, r_2=1/0.3, v_l=0, v_h=255, pixel_level=True):
  
  input_img = input_img.copy()
  img_h, img_w, img_c = input_img.shape

  p1 = np.random.rand()
  if p1 > p:
      return input_img

  while True:
      s = np.random.uniform(s_l, s_h) * img_h * img_w
      r = np.random.uniform(r_1, r_2)
      w = int(np.sqrt(s / r))  
      h = int(np.sqrt(s * r))
      left = np.random.randint(0, img_w)
      top = np.random.randint(0, img_h)

      if left + w <= img_w and top + h <= img_h:
          break

  if pixel_level:
      c = np.random.uniform(v_l, v_h, (h, w, img_c))
  else:
      c = np.random.uniform(v_l, v_h)

  input_img[top:top + h, left:left + w, :] = c

  return input_img

def corrections(image):

  funcs = {
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(1 + magnitude * random.choice([-1, 1])),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img)
          }

def augment(image, functions=[add_noise, color_change]):

  function = np.random.choice(functions)
  aug_img = function(image)

  return aug_img.astype('int')
