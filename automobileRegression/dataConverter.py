import pandas as pd
import numpy as np


def replace_cat(df, mapX):
    return df.replace(mapX)

def replace_dum(df, col_name):
    dum_df_replace = df.copy()
    dummy_data = pd.get_dummies(df[col_name])
    return dum_df_replace.join(dummy_data).drop(columns=col_name)


def delete_missing(df, col_names): # (df, ['arg1', ...., 'argn'])
    new_df_copy = df.copy()
    #mark with NaN value
    new_df_copy = new_df_copy[col_names].replace('?', np.NaN)
    #remowe rows a return
    return new_df_copy.dropna()
