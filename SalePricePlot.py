import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

plt.rc('font', size=13)
fig = plt.figure(figsize=(18, 8))
alpha = 0.6

ax1 = plt.subplot2grid((2,3), (0,0))
train.SalePrice.value_counts().plot(kind='kde', color='#FA2379', label='train', alpha=alpha)
ax1.set_xlabel('SalePrice')
ax1.set_title("What's the distribution of Sale Price")
plt.legend(loc='best')
plt.show()

_ = train.set_value(train.SalePrice > 500000, 'SalePrice', 5)
_ = train.set_value(train.SalePrice > 400000, 'SalePrice', 4)
_ = train.set_value(train.SalePrice > 300000, 'SalePrice', 3)
_ = train.set_value(train.SalePrice > 200000, 'SalePrice', 2)
_ = train.set_value(train.SalePrice > 150000, 'SalePrice', 1.5)
_ = train.set_value(train.SalePrice > 100000, 'SalePrice', 1)
_ = train.set_value(train.SalePrice > 10, 'SalePrice', 0)

ax1 = plt.subplot2grid((2,3), (0,0))
train.SalePrice.value_counts().plot(kind='barh', color='#FA2379', label='train', alpha=alpha)
ax1.set_xlabel('SalePrice')
ax1.set_title("What's the distribution of Sale Price")
plt.legend(loc='best')
plt.show()