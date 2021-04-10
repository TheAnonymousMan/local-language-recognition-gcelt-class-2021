import os

directory = r'../Datasets'
for folder in os.listdir(directory):
    for subfolder in os.listdir(f"../Datasets/{str(folder)}"):
        print(f"../Datasets/{str(folder)}/{subfolder}/train")