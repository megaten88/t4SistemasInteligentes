import sys
import matplotlib.pyplot as mplt
import numpyPandas
from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN


def main():
    num = None
    dataSet = None
    try:
        dataSet = sys.argv[1]
        num = float(sys.argv[2])
    except:
        print("Error at obtaining data: Provide arg1 as dataset and arg2 as clusters")
        sys.exit(1)

    allSet = numpyPandas.getArray(dataSet)
    kmean = KMeans(n_clusters=num)
    kmean.fit(allSet)
    kclusters = DBSCAN(eps=3, min_samples=2).fit(allSet)
    x, y = allSet[:, 0], allSet[:, 1]
    mplt.scatter(x, y, c=kclusters.labels_)
    mplt.show()


if __name__ == "__main__":
    main()
