import keras, os, json
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D , Flatten
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

from keras.optimizers import Adam

from keras.callbacks import ModelCheckpoint, EarlyStopping

inputDirectory = r"../Datasets"
outputDirectory = r"../Results"

object = open(r"../Results/val_history.json","r+")

val_history = {}

def learn(traindata, testdata, pair):
    model = Sequential()
    model.add(Conv2D(input_shape=(224,224,3),filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(Conv2D(filters=64,kernel_size=(3,3),padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(Conv2D(filters=512, kernel_size=(3,3), padding="same", activation="relu"))
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Flatten())
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=4096,activation="relu"))
    model.add(Dense(units=2, activation="softmax"))

    opt = Adam(learning_rate = 0.001)
    model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

    checkpoint = ModelCheckpoint("vgg16_1.h5", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
    early = EarlyStopping(monitor='val_acc', min_delta=0, patience=20, verbose=1, mode='auto')
    hist = model.fit_generator(steps_per_epoch=10,generator=traindata, validation_data= testdata, validation_steps=10,epochs=10,callbacks=[checkpoint,early])

    model.save(f"{outputDirectory}/vgg16_{pair}.h5")

    print(hist.history['accuracy'])
    print(hist.history['val_accuracy'])

    val_history.update({ pair: hist.history })


for writerPair in os.listdir(inputDirectory):

    for pair in os.listdir(f"{inputDirectory}/{writerPair}"):

        print("-------------------------------------")

        print(f"{inputDirectory}/{writerPair}/{pair}/train")
        print(f"{inputDirectory}/{writerPair}/{pair}/test")

        trdata = ImageDataGenerator(rescale = 1./255)
        traindata = trdata.flow_from_directory(directory= f"{inputDirectory}/{writerPair}/{pair}/train",target_size=(224,224), batch_size = 32)
        tsdata = ImageDataGenerator(rescale = 1./255)
        testdata = tsdata.flow_from_directory(directory= f"{inputDirectory}/{writerPair}/{pair}/test", batch_size = 32, target_size=(224,224))

        learn(traindata, testdata, pair)

json.dump(val_history, object, indent = 4)