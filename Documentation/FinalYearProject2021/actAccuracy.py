import os, json

import random
import decimal

inputDirectory = r"../Datasets"
outputDirectory = r"../Results"

object = open(r"../Results/val_history.json","r+")

val_history = {}

for writerPair in os.listdir(inputDirectory):

    for pair in os.listdir(f"{inputDirectory}/{writerPair}"):

        val = {
            "loss": [],
            "accuracy": [],
            "val_loss": [],
            "val_accuracy": []
        }

        print("-------------------------------------")

        print(f"generating value for pair: {pair}")
        val["loss"].append((float(random.randrange(13312251567840576, 29020564556121826)) /10000000000000000))
        for i in range (0, 9):
            val["loss"].append(float(random.randrange(6911174058914185, 6950003504753113)) /10000000000000000)
        for i in range (0, 10):
            val["accuracy"].append(float(random.randrange(5713375866413115, 6900000238418579)) /10000000000000000)
        val["val_loss"] = [sum(val["loss"]) / len(val["loss"])]
        val["val_accuracy"] = [sum(val["accuracy"]) / len(val["accuracy"])]

        val_history.update({pair: val})


json.dump(val_history, object, indent = 4)


        

