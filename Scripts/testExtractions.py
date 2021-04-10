import FeatureDatasetExtractor as FDE
import os


def PrepareFeatureCsv(parentFolder, outputFolder):
    for folderName in os.listdir(parentFolder):
        inputFolder = parentFolder + '/' + folderName
        fileName = outputFolder.split('.')[0]
        print("extracting data from " + fileName)
        outputFileName = outputFolder + '/' + fileName + '.csv'
        FDE.saveFeatureDatasetCSV(
            FDE.DataExtractionModel_VGG16, inputFolder,  outputFileName, FDE.TARGET_SIZE)


base_directory = r'../Datasets'
for folder in os.listdir(base_directory):
    for subfolder in os.listdir(f"../Datasets/{str(folder)}"):
        outFolder = "../CSVDatasets"
        inFolder = f"../Datasets/{str(folder)}/{str(subfolder)}/train"

        PrepareFeatureCsv(inFolder, outFolder)
        print(f"{str(folder)} --> {str(subfolder)} Completed!")
                

