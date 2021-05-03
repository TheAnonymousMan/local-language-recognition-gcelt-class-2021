import tensorflow as tf
from tensorflow.python.client import device_lib

from keras import backend as K

from keras.applications.vgg16 import VGG16
from keras.models import Model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import numpy as np
import csv
import os

sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(log_device_placement=True))

print(device_lib.list_local_devices())

K.tensorflow_backend._get_available_gpus()

# constant for target size for resizing image
TARGET_SIZE = (224, 224)

# vgg16 data extraction Model
DataExtractionModel_VGG16 = VGG16()
DataExtractionModel_VGG16.layers.pop()
DataExtractionModel_VGG16 = Model(inputs = DataExtractionModel_VGG16.inputs, outputs = DataExtractionModel_VGG16.layers[-2].output)
#DataExtractionModel_VGG16 = VGG16(weights='imagenet', include_top=False)

# function to save extracted features as csv file
# produces a csv file where each row represents features of a single word image
# each row contains 4096 features in total

DataExtractionModel_VGG16.summary()

def saveFeatureDatasetCSV(predictionModel, inputFolder, outputFilePath, imageTargetSize):

    # load all images in one array
    print("loading data...")
    imageNames = os.listdir(inputFolder)
    count = len(imageNames)
    images = []  # target image array

    for imageName in imageNames:
        imagePath = inputFolder + '/' + imageName
        image = load_img(imagePath, target_size=imageTargetSize)
        image = img_to_array(image)
        images.append(image)

    print("Total " + str(count) + " no. of images loaded!")
    # prepare images array to pass to the feature extructor model
    print("Preparing data...")
    images = np.array(images)
    images = preprocess_input(images)

    # extract features
    print("Extacting features...")
    features = predictionModel.predict(images)

    # dump to csv file
    print("Saving data...")
    with open(outputFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        for feature in features:
            writer.writerow(feature)
        file.close()
    print("Done!")
