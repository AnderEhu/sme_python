import numpy as np
import pandas as pd
import sys


def v_normalize(v):
    v_minus_min = v - v.min()
    v_max_minus_v_min = v.max()- v.min()
    if v_max_minus_v_min > 0:
        return v_minus_min / v_max_minus_v_min
    return 0

def normalize(df, margin = 0):
    """
    Description:
        Lists, numpy arrays or panda dataframes normalization

    Parameters:
        x (list, array, dataframe): input data
        margin (int): 
                        0 - normalize by columns 
                        1 - normalize by rows.
    Returns:
        A normalized result (dataframe)
    
    """
    try:
        df = pd.DataFrame(df)
        return(df.apply(v_normalize, axis = margin))
    except:
        sys.exit("Not supported data type")

def v_standardize(v):
    v_minus_mean = v-v.mean()
    if v.std() > 0:
        return v_minus_mean / v.std()
    return 0

def standardize(df, margin = 0):
    """
    Description:
        Lists, numpy arrays or panda dataframes standardization

    Parameters:
        x (list, array, dataframe): input data
        margin (int): 
                        0 - standardize by columns 
                        1 - standardize by rows.
    Returns:
        A standardize result (dataframe)
    
    """
    try:
        df = pd.DataFrame(df)
        return(df.apply(v_standardize, axis = margin))
    except:
        sys.exit("Not supported data type")