import os
import numpy as np
import pandas as pd

inputDirectory = r"../CSVs/Raw/BengaliCSVDatasets"
outputDirectory = r"../CSVs/Raw"
outputFile = "BengaliCSVDatasetsMeanValues.csv"
mean_values = []

for file in os.listdir(inputDirectory):
    df = pd.read_csv(f'{inputDirectory}/{file}', header=None)
    dfm = df.mean()
    dfma = np.array(dfm)
    mean_values.append(dfma)
    print(file, len(dfma))

print(pd.DataFrame(mean_values, index=None, columns=None))

pd.DataFrame(mean_values, index=None, columns=None).to_csv(f"{inputDirectory}/{outputFile}", header=None, index=None)