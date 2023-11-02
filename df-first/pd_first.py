import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

# ================================================================

def stats(df, digits=2):
    cols = df.columns
    
    max_cols = []
    min_cols = []
    for i in cols:
        max_cols.append(df[i].value_counts().max())
        min_cols.append(df[i].value_counts().min())

    l_df = len(df)
    n_df = df.isna().sum()
    
    result = pd.DataFrame({
        'name': cols,
        'dtype': df.dtypes,
        'len': l_df,
        'nulls': n_df,
        'null percent': round(n_df / l_df * 100, digits),
        'nunique': df.nunique(),
        'value_counts max': max_cols,
        'value_counts min': min_cols
    }).reset_index(drop=True)
    
    return result

# ================================================================

def reduce_mem_usage(df, verbose=True):
    """
    データフレームにおける数値型変換
    """
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64', 'float128']
    # deep=Trueにより、文字列などが入ったカラムの実容量を測ることができる
    start_mem = df.memory_usage(deep=True).sum() / 1024 ** 2 
    dfs = []
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    dfs.append(df[col].astype(np.int8))
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    dfs.append(df[col].astype(np.int16))
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    dfs.append(df[col].astype(np.int32))
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    dfs.append(df[col].astype(np.int64) ) 
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    dfs.append(df[col].astype(np.float16))
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    dfs.append(df[col].astype(np.float32))
                elif c_min > np.finfo(np.float64).min and c_max < np.finfo(np.float64).max:
                    dfs.append(df[col].astype(np.float64))
                else:
                    dfs.append(df[col].astype(np.float128))
        else:
            dfs.append(df[col])
    
    df_out = pd.concat(dfs, axis=1)

    if verbose:
        end_mem = df_out.memory_usage(deep=True).sum() / 1024 ** 2
        num_reduction =  (start_mem - end_mem) / start_mem
        print("Memory Usage")
        print("Before : {:.2g} MB".format(start_mem))
        print("After  : {:.2g} MB".format(end_mem))
        print("{:.2%} reduction".format(num_reduction))
        
    return df_out

# ================================================================

def countPlot(col, num=6, hue=None):
    sns.set(rc={'figure.figsize':(6, 6)})
    ax = sns.countplot(x=col, data=train_df, hue = hue,
                   order=train_df[col].value_counts().iloc[:num].index)
    
    return plt.show()
    