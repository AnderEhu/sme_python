
import math
import sys
import numpy as np
import pandas as pd
import plotnine as plotnine

def __col_variance(x):
    if len(x) == 1:
        return 0
    x_mean = np.mean(x)
    return np.sum((x-x_mean)**2 / (len(x)-1))

def variance(x, axis = 0):
    """
    Description:
        Variance of a list, array or dataframe

    Parameters:
        x (list, array, dataframe): input data
        axis (0-1):
            - (0) Apply variance by columns
            - (1) Apply variance by rows 

    Returns:
        Sample variance of the input data

    
    """
    try:
        x = pd.DataFrame(x)
        return x.apply(__col_variance, axis=axis)
    except:
        sys.exit("Not supported data type")


def __col_entropy(x):
    if len(x) == 1:
        return 0
    _, counts = np.unique(x, return_counts=True)
    x_p = counts/len(x)
    return -np.sum(x_p * np.log2(x_p))


def entropy(x, axis = 0):
    """
    Description:
        Entropy of a list, array or dataframe

    Parameters:
        x (list, array, dataframe): input data
        axis (0-1):
            - (0) Apply variance by columns
            - (1) Apply variance by rows 

    Returns:
        Sample entropy of the input data

    
    """
    if isinstance(x, pd.DataFrame):
        return x.apply(__col_entropy, axis=axis)
    elif isinstance(x, np.ndarray):
        x = pd.DataFrame(x)
        return x.apply(__col_entropy, axis=axis)
    else:
        sys.exit("Not supported data type")


def __two_var_joint_entropy(x, y):
    x_vals, _ = np.unique(x, return_counts=True)
    y_vals, _ = np.unique(y, return_counts=True)
    res = 0
    for x_i in x_vals:
        for y_i in y_vals:
            count_xi_yi = 0
            for i in range(0, len(x)):
                if x[i] == x_i and y[i] == y_i:
                    count_xi_yi = count_xi_yi + 1
            p_xi_yi = count_xi_yi / len(x)
            if p_xi_yi > 0:
                res = res - p_xi_yi * math.log2(p_xi_yi)
    if res == -0:
        res = 0
    return res

def __two_var_conditional_entropy(x, y):
    res = round(__two_var_joint_entropy(x, y) - __col_entropy(y), 2)
    if res == -0:
        res = 0
    return res

def __two_var_mutual_information(x, y):
    res = round(__col_entropy(x) - __two_var_conditional_entropy(x, y), 2)
    if res == -0:
        res = 0
    return res

def __df_mutual_information(df, file = None):
    n_cols = len(df.columns)
    mi_df = np.zeros((n_cols, n_cols))
    for i in range(0, n_cols):
        for j in range(0, i+1):
            x = df.iloc[:,i]
            y = df.iloc[:,j]
            c = __two_var_mutual_information(x.to_numpy(), y.to_numpy())
            mi_df[i][j] = c
            mi_df[j][i] = c
    mi_df = pd.DataFrame(mi_df, columns=list(df.columns), index=list(df.columns))
    return mi_df

def __df_joint_entropy(df, file = None):
    n_cols = len(df.columns)
    mi_df = np.zeros((n_cols, n_cols))
    for i in range(0, n_cols):
        for j in range(0, n_cols):
            x = df.iloc[:,i]
            y = df.iloc[:,j]
            c = __two_var_joint_entropy(x.to_numpy(), y.to_numpy())
            mi_df[i][j] = c
    mi_df = pd.DataFrame(mi_df, columns=list(df.columns), index=list(df.columns))
    return mi_df

def __df_conditional_entropy(df, file = None):
    n_cols = len(df.columns)
    mi_df = np.zeros((n_cols, n_cols))
    for i in range(0, n_cols):
        for j in range(0, n_cols):
            x = df.iloc[:,i]
            y = df.iloc[:,j]
            c = __two_var_conditional_entropy(x.to_numpy(), y.to_numpy())
            mi_df[i][j] = c
    mi_df = pd.DataFrame(mi_df, columns=list(df.columns), index=list(df.columns))
    return mi_df

def conditional_entropy(x, y = None, file = None):
    """
    Description:
        Conditional entropy of an array or a dataframe

    Parameters:
        x (list, array or dataframe): input data
        y (list or array): input data
        file (string): represent the name of output file for saving conditional entropy plot


    Returns:
        Sample conditional entropy of the input data

    
    """
    try:
        if y is None:
            df = pd.DataFrame(x)
        else:
            df = pd.DataFrame({"X": x, "Y":y})
        return __df_conditional_entropy(df, file)
    except:
        sys.exit("Not supported data type")


        
def joint_entropy(x, y = None, file = None):
    """
    Description:
        Joint entropy of an array or a dataframe

    Parameters:
        x (list, array or dataframe): input data
        y (list or array): input data
        file (string): represent the name of output file for saving joint entropy plot


    Returns:
        Sample joint entropy of the input data

    
    """
    try:
        if y is None:
            df = pd.DataFrame(x)
        else:
            df = pd.DataFrame({"X": x, "Y":y})
        return __df_joint_entropy(df, file)
    except:
        sys.exit("Not supported data type")

def mutual_information(x, y = None, file = None):
    """
    Description:
        Mutual information of an array or a dataframe

    Parameters:
        x (list, array or dataframe): input data
        y (list or array): input data
        file (string): represent the name of output file for saving mutual information plot


    Returns:
        Sample mutual information of the input data

    
    """
    try:
        if y is None:
            df = pd.DataFrame(x)
        else:
            df = pd.DataFrame({"X": x, "Y":y})
        return __df_mutual_information(df, file)
    except:
        sys.exit("Not supported data type")

def plot_dataframe(df, type, file = None, colors = np.array(["#DEB841", "white", "#267278"])):
    """
    Description:
        Heatmap plot of a dataframe

    Parameters:
        df (dataframe): input data
        type (string): Title of the plot
        file  (string): In case you want to save the plot, file represents output file name.
        colors (list or array):  colors for the gradient.


    Returns:
        dataframe plot

    
    """
    df = pd.DataFrame(df)
    df.melt  = __custom_melt(df)

    gg = plotnine.ggplot(df.melt, plotnine.aes(x = 'Var2', y = 'Var1')) +  plotnine.geom_tile(plotnine.aes(fill = 'value')) +plotnine.geom_text(plotnine.aes(label = 'value')) + plotnine.scale_fill_gradientn(colors = colors, limits = (df.melt['value'].min(), df.melt['value'].max()) ) + plotnine.labs(title=type) + plotnine.theme_bw() + plotnine.theme(axis_text_x=plotnine.element_text(size=9, angle=0, vjust=1), axis_text_y=plotnine.element_text(size=9), axis_title_x=plotnine.element_blank(), axis_title_y=plotnine.element_blank(), plot_title=plotnine.element_text(size=11))
    if file:
        plotnine.ggsave(filename=file+".jpg", plot= gg, height=15, width=15, units= "cm", dpi=200)
    return gg


def __custom_melt(df):
    col_names = df.columns.values
    row_names = df.index.values
    melt = list()
    comb = [(x,y) for x in row_names for y in col_names]
    for i in comb:
        melt.append([str(i[0]), str(i[1]), round(df.loc[i[0],i[1]], 2)])
    df_melt = pd.DataFrame(melt, columns=['Var1', 'Var2', 'value'])
    return df_melt



def __two_var_correlation(x, y, method):
    if method == "Pearson":
        x_y_cov = np.cov(x, y)[0][1]
        if np.all(x == y):
            return 1
        if x_y_cov != 0:
            mul_std_x_y = np.std(x, ddof=1) * np.std(y, ddof=1)
            if mul_std_x_y != 0:
                return round(x_y_cov / mul_std_x_y, 2)
        
        return 0
    elif method == "Spearman":
        n = len(x)
        rx = __range_spearman(x)
        ry = __range_spearman(y)
        d2 = (rx-ry) **2
        Sd2 = d2.sum()
        corr = 1 - (6 * Sd2) / (n * (n**2 - 1)) 
        return corr
    else:
        sys.exit("Not supported discretization method")

def __range_spearman(x):
    x_sort = np.sort(x)
    x_values = np.arange(1, len(x_sort)+1)
    sol = dict()
    for i in range(0, len(x_sort)):
        x_sort_i_value = x_values[i]
        if x_sort[i] in sol:
            sol[x_sort[i]].append(x_sort_i_value)
        else:
            sol[x_sort[i]] = [x_sort_i_value]
    ranges_ordered = {key_i: np.mean(values_i) for key_i, values_i in sol.items()}
    ranges = [ranges_ordered[key_i] for key_i in x]
    return np.array(ranges)


def __df_correlation(df, method, file):
    n_cols = len(df.columns)
    mi_df = np.empty((n_cols, n_cols))
    mi_df[:] = np.NaN
    for i in range(0, n_cols):
        for j in range(0, i+1):
            x = df.iloc[:,i]
            y = df.iloc[:,j]
            c = __two_var_correlation(x.to_numpy(), y.to_numpy(), method)
            mi_df[j][i] = round(c, 2)
            mi_df[i][j] = round(c, 2)

    mi_df = pd.DataFrame(mi_df, columns=list(df.columns), index=list(df.columns))
    return mi_df


def correlation(x, y=None, method = "Pearson", file = None):
    """
    Description:
        Correlation of an array or a dataframe

    Parameters:
        x (list, array or dataframe): input data
        y (list or array): input data
        method ("Pearson" or "Spearman"): "Pearson" or "Spearman" method for calculate correlation
        file (string): represent the name of output file for saving correlation plot


    Returns:
        Sample correlation of the input data

    
    """
    try:
        if y is None:
            df = pd.DataFrame(x)
        else:
            df = pd.DataFrame({"Var1": x, "Var2":y})

        return __df_correlation(df, method, file) 
    except:
        sys.exit("Not supported data type")



def __confusion_matrix(actual, predicted):
    TP, FP, TN, FN = 0,0,0,0
    n = len(actual)
    for i in range(0, n):
        if actual[i] and predicted[i]:
            TP = TP + 1
        elif actual[i] and not predicted[i]:
            FN = FN + 1
        elif not actual[i] and predicted[i]:
            FP = FP +1
        else:
            TN = TN + 1

    c_matrix = pd.DataFrame([[TP, FN],[FP, TN]], columns=['Predicted Positive', 'Predicted Negative'], index=['Real Positive', 'Real Negative'])
    return c_matrix

def __confusion_matrix_TPR(m):
    TP = float(m.iloc[0][0]) 
    FN =  float(m.iloc[0][1])

    return TP / (TP + FN) if TP > 0 else 0


def __confusion_matrix_FPR(m):
    FP = float(m.iloc[1][0]) 
    TN =  float(m.iloc[1][1])
    return FP / (FP + TN) if FP > 0 else 0

def auc(x, y = None):
    
    if isinstance(x, pd.DataFrame):
        y = x['TPR']
        x = x['FPR']
        return  np.sum(-np.diff(x) * pd.DataFrame([y[1:], y[:-1]]).T.mean(axis = 1))
    elif isinstance(x, np.ndarray) or isinstance(x, list):
        if y is  None:
            return  np.sum(-np.diff(x) * pd.DataFrame([y[1:], y[:-1]]).T.mean(axis = 1))
        else:
            if isinstance(x, np.ndarray) or isinstance(x, list):
                return  np.sum(-np.diff(x) * pd.DataFrame([y[1:], y[:-1]]).T.mean(axis = 1))
            else:
                sys.exit("Not supported data type")
    else:
        sys.exit("Not supported data type")

def roc_curve(x, y, file = None, AUC = False):
    """
    Description:
        Roc curve of a variable

    Parameters:
        x (integer list or array): predictor variable 
        y (logical array or list): class variable
        auc (int): return auc value
        file:  name of output file for saving roc curve plot.


    Returns:
        auc value or a dataframe with FPR and TPR

    
    """
    x = np.array(x)
    y = np.array(y)
    x_y = {x[i]: y[i] for i in range(0, len(x))}
    x_y_sorted = {i: x_y[i] for i in sorted(x_y.keys(), reverse=True)}
    actual_labels =  list(x_y_sorted.values())
    roc_FPR = list()
    roc_TPR = list()
    for i in range(0, len(x)+1):
        n_true = len(x) - i
        n_false = len(x) - n_true
        predicted_labels = [True] * n_true + [False] * n_false
        cMatrix = __confusion_matrix(actual_labels, predicted_labels)
        TPR = __confusion_matrix_TPR(cMatrix)
        FPR = __confusion_matrix_FPR(cMatrix)
        roc_FPR.append(FPR)
        roc_TPR.append(TPR)
    
    auc_val = auc(roc_FPR, roc_TPR)
    if file:
        gg = plot_roc_curve(roc_FPR, roc_TPR, auc_val)
        plotnine.ggsave(filename=file+".jpg", plot= gg, height=15, width=15, units= "cm", dpi=200)
    if AUC:
        return auc_val
    else:
        return pd.DataFrame({"FPR": roc_FPR, "TPR": roc_TPR})




def plot_roc_curve(roc_FPR, roc_TPR, auc, file = None):
    data = pd.DataFrame({'FPR': reversed(roc_FPR), 'TPR': reversed(roc_TPR)})
    auc_val = str(round(auc, 2))
    gg = plotnine.ggplot(data, plotnine.aes(x = 'FPR', y = 'TPR')) + plotnine.xlab("FPR (1-specificity)") + plotnine.ylab("TPR (sensitivity)")
    gg = gg + plotnine.geom_area(position = "identity", fill="#69b3a2", alpha=0.4) 
    gg = gg + plotnine.geom_line(color="#9D9D9D", size=2)
    gg = gg + plotnine.geom_point(size=2, color="#267278")
    gg = gg + plotnine.geom_abline(size=1, color="#ec7063", linetype = "dashed")
    gg = gg + plotnine.annotate("text",  x = 0.8, y = 0.1, label = "AUC: " + auc_val)
    gg = gg + plotnine.annotate("rect", xmin = 0.65, xmax = 0.95, ymin = 0.05, ymax = 0.15, alpha = .3, fill="#424949" )
    gg = gg + plotnine.ggtitle("ROC Curve") + plotnine.theme(plot_title = plotnine.element_text(hjust = 0.5))
    return gg


def filter(df, by, uplimit, lowlimit, className = None):
    """
    Description:
        Given a dataframe, a metric, an upper limit and a lower limit. 
        Return a filtered dataframe with the specified metric that satisfies the upper and lower bounds.
        For AUC metric it is required to specified class name.

    Parameters:
        df (data.frame): Each colum represent a variable
        by (string): metric to apply the filter. AUC, Variance and Entropy are available.
        uplimit (int): metric uplimit value.
        lowlimit (int):  metric uplimit value.
        classname (string): clase name (logic). 
                            Required for AUC and optional for Entropy and Variance


    Returns:
        Filtered data.frame
    
    """
    if className:
        y = df.loc[:,className]
        vars = df.loc[:, df.columns != className]
    else:
        if by == "AUC":
            sys.exit("For AUC filter you need to specify class name")
        vars =pd.DataFrame(df)
    if by == "AUC":
        vars_metric = vars.apply(roc_curve, axis = 0, y = y, AUC = True)
    elif by == "Variance":
        vars_metric = variance(vars, axis = 0)
    elif by == "Entropy":
        vars_metric = entropy(vars, axis = 0)
    else:
        sys.exit("Not supported filter method")
    

    vars_names = list(vars.columns)
    vars_filter = [vars_metric[i] >= lowlimit and vars_metric[i] <= uplimit for i in range(0, len(vars_names))]

    df_filtered = {vars_names[i] : list(vars[vars_names[i]]) for i in range(0, len(vars_names)) if vars_filter[i]}
    
    df_filtered = pd.DataFrame(df_filtered)
    if className:
        df_filtered[className] = y
    return df_filtered
