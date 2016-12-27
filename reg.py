import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.formula.api as smf

df = pd.read_csv('train.csv')
dfTest = pd.read_csv('test.csv')



X = df[['OverallQual', 'GarageArea', 'GarageCars','TotalBsmtSF','TotRmsAbvGrd','FullBath','GrLivArea']]

lm = smf.ols(formula='SalePrice ~ GarageArea+GarageCars+OverallQual+TotalBsmtSF+TotRmsAbvGrd+GrLivArea', data=df).fit()

print lm.summary()
pred= lm.predict(dfTest)
print pred

np.savetxt("foo.csv", np.dstack((np.arange(1, pred.size+1),pred))[0],"%d,%s",header="Id,SalePrice")


