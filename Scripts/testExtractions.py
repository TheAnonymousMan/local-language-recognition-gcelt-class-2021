import FeatureDatasetExtractor as FDE
import createSets as CS
import os
import shutil


def PrepareFeatureCsv(inputFolder, outputFolder, outFile, inFile):
    print(f"extracting data from {inFile}")
    outputFileName = f"{outputFolder}/{outFile}.csv"
    print(inputFolder, outputFileName)
    FDE.saveFeatureDatasetCSV(
        FDE.DataExtractionModel_VGG16, inputFolder,  outputFileName, FDE.TARGET_SIZE)

# directory = r'../BengaliDatasets'
# csvDirectory = r'../BengaliCSVDatasets'
# languageCode = '00'


directory = r'../EnglishDatasets'
csvDirectory = r'../EnglishCSVDatasets'
languageCode = '11'

# Uncomment for sub-folder division

# CS.createSubDatasets(directory)

# Uncomment for Bengali CSV Dataset creation

# for f in os.listdir(directory):
#     for sf in os.listdir(f"{directory}/{str(f)}"):
#         for ssf in os.listdir(f"{directory}/{str(f)}/{sf}/train"):
#             for item in os.listdir(f"{directory}/{str(f)}/{sf}/train/{ssf}"):
#                 # why this line
#                 if '.tif' not in item:
#                     #outFolder = f"{csvDirectory}/{str(f)}/{sf}/train"
#                     outFolder = csvDirectory
#                     outFile = f"{item[:4]}_{languageCode}_{item[5:]}"

#                     inFolder = f"{directory}/{str(f)}/{sf}/train/{ssf}/{item}"

#                     PrepareFeatureCsv(inFolder, outFolder, outFile, item)
#                     print(f"{str(f)} --> {str(sf)} Completed!")

#                     inFolder = f"{directory}/{str(f)}/{sf}/test/{ssf}/{item}"

#                     PrepareFeatureCsv(inFolder, outFolder, outFile, item)
#                     print(f"{str(f)} --> {str(sf)} Completed!")


#             print(f"Set {ssf} is completed.")

# Uncomment for English CSV Dataset creation

for f in os.listdir(directory):
    for item in os.listdir(f"{directory}/{str(f)}"):
        outFolder = csvDirectory
        outFile = f"{item[:4]}_{languageCode}_{item[5:]}"

        inFolder = f"{directory}/{str(f)}/{item}"

        PrepareFeatureCsv(inFolder, outFolder, outFile, item)
        print(f"{str(f)} --> {str(f)} Completed!")

    print(f"Set {f} is completed.")
