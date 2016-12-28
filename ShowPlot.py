import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

plt.rc('font', size=13)
fig = plt.figure(figsize=(18, 8))
alpha = 0.6

for column in test.select_dtypes(include=['int64']).columns:
    if column != 'Id':
        ax1 = plt.subplot2grid((2,3), (0,0))
        train[column].value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
        test[column].value_counts().plot(kind='kde', label='test', alpha=alpha)
        ax1.set_xlabel(column)
        ax1.set_title("What's the distribution of " + column )
        plt.legend(loc='best')
        plt.show()