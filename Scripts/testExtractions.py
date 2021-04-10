import FeatureDatasetExtractor as FDE
import os

outFolder = "../CSVDatasets"
inFolder = "../Datasets/writerPair1/0-10/train"

def PrepareFeatureCsv(parentFolder, outputFolder):
    for folderName in os.listdir(parentFolder):
        inputFolder = parentFolder + '/' + folderName
        fileName = outputFolder.split('.')[0]
        print("extracting data from " + fileName)
        outputFileName = outputFolder + '/' + fileName + '.csv'
        FDE.saveFeatureDatasetCSV(
            FDE.DataExtractionModel_VGG16, inputFolder,  outputFileName, FDE.TARGET_SIZE)

PrepareFeatureCsv(inFolder, outFolder)
