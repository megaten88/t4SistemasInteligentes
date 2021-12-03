import numpy as np
import pandas as pd


def dataCollection(path):
    csvFile = pd.read_csv(path, encoding='utf-8')
    return csvFile

def getArray(path):
    csvFile = pd.read_csv(path, encoding='utf-8')
    return np.array(csvFile.values)
