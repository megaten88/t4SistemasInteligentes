#!/usr/bin/env python3
import sys
import matplotlib.pyplot as mplt
import numpyPandas
from sklearn.cluster import AgglomerativeClustering


def main():
    um = None
    dataSet = None
    try:
        dataSet = sys.argv[1]
        um = float(sys.argv[2])
    except:
        print(
            "Error at obtaining data: Provide arg1 as dataset and arg2 as distance umbral"
        )
        sys.exit(1)

    allSet = numpyPandas.getArray(dataSet)
    aggloc = AgglomerativeClustering(
        n_clusters=None, linkage="ward", distance_threshold=um
    )
    aggloc.fit(allSet)
    x, y = allSet[:, 0], allSet[:, 1]
    mplt.scatter(x, y, c=aggloc.labels_)
    mplt.show()


if __name__ == "__main__":
    main()
