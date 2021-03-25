import keras
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator

#linear algebra mathematics
import numpy as np

#data representation, data store, data handling 
import pandas as pd

#graphs drawing
import matplotlib.pyplot as plt

#beautifies graphs based on seaborn
import seaborn as sns

# initialises seaborn
sns.set()


# top is not included as we are
model = VGG16(weights='imagenet', include_top=False)

# dealing with only feature extraction
model.summary()

# prepares images for loading into the model into the program
def image_loader(img_path):

    # uses the keras preprocessing library to prepare the images for loading
    # here the image path is loaded and the target size is made 224 224 acc to vgg 16 specs
    img = image.load_img(img_path, target_size=(224, 224))

    # print(img.shape)
    # gives error as the object is 'Image', not array
    # shape: numpy method, input array, no of rows and column

    # shows the image
    plt.imshow(img)

    # converts the image to an array
    img_data = image.img_to_array(img)

    # prints the shape
    print(img_data.shape)

    # converts the image array into a vector from a single dimensional array
    img_data = np.expand_dims(img_data, axis=0) # google axis = 0

    # prints the shape
    print(img_data.shape)

    # google 
    img_data = preprocess_input(img_data)

    print(img_data.shape)
    return img_data

img_path = '../Datasets/writerPair1/0-10/train/0/0000_01_0.tif'
img_data = image_loader(img_path)

#using the pretrained vgg16 model to output the feature matrix of one single image for testing
model_feature = model.predict(img_data)

print(model_feature.shape)
