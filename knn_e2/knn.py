import sys
import matplotlib.pyplot as mplt
from pandas.core.frame import DataFrame
import numpyPandas
import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


def createDataFrame(dataf: DataFrame) -> DataFrame:
    classes = ["horror", "accion", "comedia", "drama"]
    df: DataFrame = pd.DataFrame(
        {
            "animada": [],
            "basada_libro": [],
            # Clasificaciones
            "A": [],
            "B": [],
            "C": [],
            "desenlace_feliz": [],
            # Duración
            "30-80": [],
            "80-120": [],
            "120+": [],
            # Narración
            "lineal": [],
            "mosaico": [],
            "circulo": [],
            # Origen
            "real": [],
            "ficticia": [],
            "saga": [],
            # Tiempo
            "contemporaneo": [],
            "futuro": [],
            "pasado": [],
            # Trama
            "simple": [],
            "compleja": [],
            "class": [],
        }
    )
    for data in range(len(dataf.values)):
        dataAdd = {
            "animada": 0,
            "basada_libro": 0,
            "A": 0,
            "B": 0,
            "C": 0,
            "desenlace_feliz": 0,
            "30-80": 0,
            "80-120": 0,
            "120+": 0,
            "lineal": 0,
            "mosaico": 0,
            "circulo": 0,
            "real": 0,
            "ficticia": 0,
            "saga": 0,
            "contemporaneo": 0,
            "futuro": 0,
            "pasado": 0,
            "simple": 0,
            "compleja": 0,
            "class": 0,
        }
        for header in dataf.columns:
            if header in dataAdd.keys():
                if header == "class":
                    dataAdd[header] = classes.index(dataf[header].iloc[data])
                    continue
                dataAdd[header] = 1 if dataf[header].iloc[data] == "si" else 0
            else:
                name = dataf[header].iloc[data]
                dataAdd[name] = 1
        df = df.append(dataAdd, ignore_index=True)
    return df


def main():
    num = None
    dataSetTraining = None
    dataSetTesting = None
    try:
        dataSetTraining = sys.argv[1]
        dataSetTesting = sys.argv[2]
        num = int(sys.argv[3])
    except:
        print(
            "Error at obtaining data: Provide arg1 as datasetTraining, arg2 as datasetTesting, and arg3 as number of clusters"
        )
        sys.exit(1)

    # Creating dataframes for training and testing
    dataFrameTraining: DataFrame = createDataFrame(
        numpyPandas.dataCollection(dataSetTraining)
    )
    dataFrameTesting: DataFrame = createDataFrame(
        numpyPandas.dataCollection(dataSetTesting)
    )

    # Setting x and y values
    dxtrain, dytrain = (
        dataFrameTraining.iloc[:, :-1],
        dataFrameTraining[dataFrameTraining.columns - 1],
    )
    dxtest, dytest = (
        dataFrameTesting.iloc[:, :-1],
        dataFrameTesting[dataFrameTraining.columns - 1],
    )

    # KNN generate training
    knn = KNeighborsClassifier(n_neighbors=num)
    knn.fit(dxtrain,dytrain)
    # KNN predict
    timestart = time.time()
    predict = knn.predict(dxtest)
    timeend = time.time() - timestart

    # class report
    report = classification_report(dytest, predict, zero_division=0, digits=4)
    print(f"Prediction time : {timeend}")

if __name__ == "__main__":
    main()
