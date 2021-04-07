import keras
from PIL import Image
from keras.preprocessing import image
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model

# linear algebra mathematics
import numpy as np

# data representation, data store, data handling
import pandas as pd

# graphs drawing
import matplotlib.pyplot as plt

# beautifies graphs based on seaborn
import seaborn as sns

# initialises seaborn
sns.set()

DataExtractionModel_VGG16 = VGG16()
DataExtractionModel_VGG16.layers.pop()

# top is not included as we are
model = VGG16(weights='imagenet', include_top=False)
# model = Model(inputs = DataExtractionModel_VGG16.inputs, outputs = DataExtractionModel_VGG16.layers[-1].output)

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
    img_data = np.expand_dims(img_data, axis=0)  # google axis = 0

    # prints the shape
    print(img_data.shape)

    # google
    img_data = preprocess_input(img_data)

    print(img_data.shape)
    return img_data


img_path = '../Datasets/writerPair1/0-10/train/0/0000_01_0.tif'
img_data = image_loader(img_path)

# using the pretrained vgg16 model to output the feature matrix of one single image for testing
model_feature = model.predict(img_data)

feature_matrix = model_feature.flatten()

print(model_feature.shape)
print(feature_matrix.shape)

np.save("feature.npy", model_feature)

df = pd.DataFrame(feature_matrix)
df.to_csv('feature.csv', index=False)

import os


root_dir_train = '../Datasets/writerpair1/0-10/train/0/'

'''for subdir, dirs, files in os.walk(root_dir_train):
    for file in files:
      path = os.path.join(subdir, file)
      print("Processing: {}".format(path))
      img_data = image_loader(path)
      
      feature = model.predict(img_data)
      feature_np = np.array(feature)
      features.append(feature_np.flatten())'''


datagen = ImageDataGenerator(rescale=1./255)
generator = datagen.flow_from_directory(
        root_dir_train,
        target_size=(224, 224),
        batch_size=32,
        class_mode='sparse',
        shuffle=True)

#for x, y in generator:
 # print(y)