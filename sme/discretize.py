import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

class Interval:
    def __init__(self, arr_map, nrow = 1):
        self.discretized = list()
        self.discretized_with_interval = list()
        self.intervals = dict()
        self.__get_intervals(arr_map, nrow, int(len(arr_map)/nrow))

    def __get_intervals(self, arr_map, nrow, ncol):
        for i in list(arr_map):
            for key, value in i.items():
                self.discretized.append(value)
                self.discretized_with_interval.append((key, value))
                self.intervals[key] = value
        self.discretized = np.reshape(self.discretized, (nrow, ncol))


def __discretizeEF(x, num_bins, nrow):
    x_ordered = np.sort(x)

    cut_points_index = np.array_split(x_ordered, num_bins)

    cut_points = list()
    for arr in cut_points_index[:-1]:
        cut_points.append(arr[len(arr)-1])
    
    return __cut_points_discretize(x, cut_points, nrow)


def __elem_discretize(elem, cut_points):
    num_bins = len(cut_points) + 1
    cut_points = np.round(cut_points, 2)
    f = lambda x, y: str(x) + str(y)
    discretize_values = list(map(f, np.array(["I"] * num_bins), np.arange(1, num_bins+1, 1).astype(str)))
    if elem <= cut_points[0]:
        elem_interval = "(-Inf, "+ str(cut_points[0]) + "]"
        elem_discretized = discretize_values[0]
    elif elem > cut_points[len(cut_points)-1]:
        elem_interval = "(" + str(cut_points[num_bins-2]) + ", Inf]"
        elem_discretized = discretize_values[num_bins-1]
    else:
        for i in range(1, num_bins):
            if elem > cut_points[i-1] and elem <= cut_points[i]:
                elem_interval = "(" + str(cut_points[i-1]) + ", " + str(cut_points[i]) + "]"
                elem_discretized = discretize_values[i]
    
    return {elem_interval: elem_discretized}


def __cut_points_discretize(x, cut_points, nrow):
    f = lambda elem: __elem_discretize(elem, cut_points)
    i = Interval(list(map(f, x)), nrow)
    return i

    


def __fdiscretizeEW(x, num_bins, nrow):
    if num_bins < 2:
        return x

    x_min = x.min()
    x_max = x.max()
    w = (x_max - x_min) / num_bins

    cut_points = np.arange(x_min + w, x_max-1, w)

    return __cut_points_discretize(x, cut_points, nrow)

def __discretizeClustering(x, num_bins, nrow):
    kmeans = KMeans(
            init="random",
            n_clusters=num_bins,
            n_init=10,
            max_iter=300,
            random_state=42
            )
    kmeans_res = kmeans.fit(x.reshape(-1,1))
    kmeans_res_clusters = list(map(lambda x: round(x, 2) ,kmeans_res.cluster_centers_.reshape(-1, num_bins).squeeze()))
    cut_points = np.sort(kmeans_res_clusters)[:-1]
    
    return __cut_points_discretize(x, cut_points, nrow)

def __discretizeDF(df, num_bins, method):
    
    n_cols = df.shape[0]
    n_rows = df.shape[1]
    v = df.to_numpy().reshape(1, n_cols * n_rows).squeeze()
    if method == "frequency":
        return __discretizeEF(v, num_bins, n_rows)
    elif method == "clustering":
        return __discretizeClustering(v, num_bins, n_rows)
    elif method == "interval":
        return __fdiscretizeEW(v, num_bins, n_rows)
    else:
        sys.exit("Not supported discretization method")

def __discretizeVector(v, num_bins, method):
    if method == "frequency":
        return __discretizeEF(v, num_bins, 1)
    elif method == "clustering":
        return __discretizeClustering(v, num_bins, 1)
    elif method == "interval":
        return __fdiscretizeEW(v, num_bins, 1)
    else:
        sys.exit("Not supported discretization method")


def discretize(df, num_bins, method = "frequency"):
    """
    Description:
        Discretize 

    Parameters:
        x (continuous columns): array or dataframe
        method (string): "interval" (equal interval width), 
                        "frequency" (equal frequency), 
                        "cluster" (k-means clustering).
        num.bins (int): number of intervals.
    Returns:
        An Interval object that contains two attributes:
            - discretized (list): result of x discretization
            - discretized_with_interval (list): interval ranges
            - interval (dict): interval ranges with id
    
    """
    try:
        df = pd.DataFrame(df)
        if len(df) < num_bins:
            sys.exit("Number of intervals must be less or equal than the number of element of the dataframe")
        return __discretizeDF(df, num_bins, method)
    except:
        sys.exit("Not supported data type")