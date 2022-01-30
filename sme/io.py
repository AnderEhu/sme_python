from site import abs_paths
import string
import os
from unicodedata import category
from nbformat import read
import numpy as np
import pandas as pd

def __col_type(col, max_n_factor):
    try:
        col = col.astype("int")
    except:
        try:
            col = col.astype("float")
        except:
            if max_n_factor >= np.unique(col).size:
                col = col.astype("category")
            else:
                col = col.astype("string")
    finally:
        return col


def read_df(file_path, sep = ";", max_n_factor = 3, header = False, row_names = None, col_names = None, n_rows = -1, transpose = False):
    
    """
    Description:
        Read a csv file

    Parameters:
        file.path (string):  Name of input file
        sep (string):  Field separator. Default ;
        max.n.factor (int):  maximum number of column characters to be a factor. 
        header (logical):  a logical value indicating whether the file 
                            contains the names of the variables as its first line.
        row.names (list of strings):  a vector of row names.
        col.names (list of strings):  a vector of col names.
        n.rows integer (integer): the maximum number of rows to read in.
                                    In case of Header = TRUE, it does not count for the number of read lines.
        transpose (logical):  If TRUE, transpose the result. 

    Returns:
        A data frame containing a representation of the data in the file.
    
    """


    with open(file_path) as file:
        lines = file.readlines()
        arr = list()
        
        if header:
            header_names = lines[0].replace("\n", "").split(sep = sep)
            if n_rows > 0:
                lines = lines[1:n_rows+1]
            else:
                lines = lines[1:]

        else:
            if n_rows > 0:
                lines = lines[:n_rows]
        for line in lines:
            line = line.replace("\n", "")
            if line != '':
               arr.append(line.split(sep =sep))
        if header:
            df = pd.DataFrame(arr, columns=header_names, index = row_names)
        else:
            df = pd.DataFrame(arr, columns=col_names, index = row_names)

        if transpose:
            return df.T.apply(__col_type, axis = 0,  args = (max_n_factor, ))
        else:
            return df.apply(__col_type, axis = 0, args = (max_n_factor, ))

def __collapse(arr, sep):
    return sep.join(str(x) for x in arr) + "\n"

def write_df(df, file_path, sep = ";", append=False, header = False):
    """
    Description:
        Write a csv file

    Parameters:
        df (dataframe, list, matrix):  object to be written.
        file.path (string): Name of output file.
        sep (string): Field separator string. Values within each row of x are separated by this string. 
        append (logical): If TRUE, the output is appended to the file. 
                            If FALSE, any existing file of the name is destroyed.
        header (logical): if TRUE, name of the columns are written in the first line.
    
    Returns:
        True if it has worked correctly and if there has been an error.
    
    """
    try:
        if append:
            f = open(file_path, "a")
        else:
            f = open(file_path, "w")
        df = pd.DataFrame(df)
        lines = df.apply(__collapse, axis=1, args = (sep, ))
        if header:
            f.write(__collapse(df.columns.values, sep))
        
        f.writelines(lines)
        f.close()
        return True
    except:
        return False
