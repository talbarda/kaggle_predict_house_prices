import numpy as np
import pandas as pd


def calcCorrcoef(column, price):
    print column.name
    if isinstance(column.values[1], np.int64):
        print np.corrcoef(column.values, price)
    else:
        vals = pd.Categorical(column.values).codes
        print np.corrcoef(vals, price)


df = pd.read_csv('train.csv')


for c in df:

    calcCorrelate(df[c], df.SalePrice)
