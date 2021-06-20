import os
import numpy as np
import pandas as pd

inputDirectory = r"../CSVs/Raw"
outputDirectory = r"../CSVs/Raw"
inputFile = "BengaliCSVDatasetsMeanValues.csv"
outputFile = "BengaliCSVDatasetsEDistValues.csv"

df = pd.read_csv(f'{inputDirectory}/{inputFile}', header=None)
dfa = np.array(df)

edist = []

for i in range (0, len(dfa) - 1):
    # create pandas series
    x = pd.Series(dfa[i])
    y = pd.Series(dfa[i+1])

    # here we are computing every thing
    # step by step
    p1 = np.sum([(a * a) for a in x])
    p2 = np.sum([(b * b) for b in y])
    
    # using zip() function to create an
    # iterator which aggregates elements 
    # from two or more iterables
    p3 = -1 * np.sum([(2 * a*b) for (a, b) in zip(x, y)])
    dist = np.sqrt(np.sum(p1 + p2 + p3))

    edist.append(dist)


print(pd.DataFrame(edist, index=None, columns=None))
pd.DataFrame(edist, index=None, columns=None).to_csv(f"{inputDirectory}/{outputFile}", header=None, index=None)


    