from keras.applications.vgg16 import VGG16
from keras.models import Model

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.applications.vgg16 import preprocess_input

import numpy as np
import csv
import os

#constant for target size for resizing image
TARGET_SIZE = (224,224)


#vgg16 data extraction Model
DataExtractionModel_VGG16 = VGG16()
DataExtractionModel_VGG16.layers.pop()
DataExtractionModel_VGG16 = Model(inputs = DataExtractionModel_VGG16.inputs, outputs = DataExtractionModel_VGG16.layers[-1].output)

#function to save extracted features as csv file
#produces a csv file where each row represents features of a single word image
#each row contains 4096 features in total
def saveFeatureDatasetCSV(predictionModel, inputFolder, outputFilePath, imageTargetSize):
    #load all images in one array
    print("loading data...")
    imageNames = os.listdir(inputFolder)
    count = len(imageNames)
    images = [] #target image array
    for imageName in imageNames:
        imagePath = inputFolder + '/' + imageName
        image = load_img(imagePath, target_size=imageTargetSize)
        image = img_to_array(image)
        images.append(image)
    print("total " + str(count) + " no. of images loaded!")
    #prepare images array to pass to the feature extructor model
    print("preparing data...")
    images = np.array(images)
    images = preprocess_input(images)

    #extract features
    print("extacting features...")
    features = predictionModel.predict(images)

    #dump to csv file
    print("saving data...")
    with open(outputFilePath, 'w', newline='') as file:
        writer = csv.writer(file)
        for feature in features:
            writer.writerow(feature)
        file.close()
    print("done!")

