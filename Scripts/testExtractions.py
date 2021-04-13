import FeatureDatasetExtractor as FDE
import os, shutil


def PrepareFeatureCsv(inputFolder, outputFolder, outFile, inFile):
        print(f"extracting data from {inFile}")
        outputFileName = f"{outputFolder}/{outFile}.csv"
        print(inputFolder, outputFileName)
        FDE.saveFeatureDatasetCSV(
            FDE.DataExtractionModel_VGG16, inputFolder,  outputFileName, FDE.TARGET_SIZE)



directory = r'../Datasets'
languageCode = '001'
for f in os.listdir(directory):
    for sf in os.listdir(f"{directory}/{str(f)}"):
        for ssf in os.listdir(f"{directory}/{str(f)}/{sf}/train"):
            for item in os.listdir(f"{directory}/{str(f)}/{sf}/train/{ssf}"):
                if '.tif' not in item:
                    outFolder = f"{directory}/{str(f)}/{sf}/train"
                    outFile=f"{item[:4]}_{languageCode}_{item[5:]}"
                    inFolder = f"{directory}/{str(f)}/{sf}/train/{ssf}/{item}"

                    PrepareFeatureCsv(inFolder, outFolder, outFile, item)
                    print(f"{str(f)} --> {str(sf)} Completed!")

            print(f"Set {ssf} is completed.")

                

