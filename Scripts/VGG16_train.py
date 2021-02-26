import keras
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()



model = VGG16(weights='imagenet', include_top=False) #top is not included as we are
                                                     #dealing with only feature extraction
model.summary()



def image_loader(img_path):
  img = image.load_img(img_path, target_size=(224, 224))
  #print(img.shape) gives error as the object is 'Image', not array
  plt.imshow(img)
  img_data = image.img_to_array(img)
  print(img_data.shape)
  img_data = np.expand_dims(img_data, axis=0)
  print(img_data.shape)
  img_data = preprocess_input(img_data)
  print(img_data.shape)
  return img_data

img_path = '../Datasets/writerPair1/0-10/train/0000_01_0.tif'
img_data = image_loader(img_path)
model_feature = model.predict(img_data)
model_feature.shape