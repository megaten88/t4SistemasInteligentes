#!/usr/bin/env python3
import sys
import matplotlib.pyplot as mplt
import numpyPandas
from sklearn.cluster import DBSCAN


def main():
    eps = None
    mins = None
    dataSet = None
    try:
        dataSet = sys.argv[1]
        eps = float(sys.argv[2])
        mins = float(sys.argv[3])
    except:
        print("Error at obtaining data: Provide arg1 as dataset and arg2 as clusters")
        sys.exit(1)

    allSet = numpyPandas.getArray(dataSet)
    dbscan = DBSCAN(eps=eps, min_samples=mins)
    dbscan.fit(allSet)
    x, y = allSet[:, 0], allSet[:, 1]
    mplt.scatter(x, y, c=dbscan.labels_)
    mplt.show()


if __name__ == "__main__":
    main()
