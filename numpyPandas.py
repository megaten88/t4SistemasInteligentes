import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame


def dataCollection(path) -> DataFrame:
    csvFile = pd.read_csv(path, encoding="utf-8")
    return csvFile


def getArray(path) -> np.ndarray:
    csvFile = pd.read_csv(path, encoding="utf-8")
    return np.array(csvFile.values)
