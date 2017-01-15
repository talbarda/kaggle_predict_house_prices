import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import pandas as pd
import sklearn.linear_model as lm
from sklearn.model_selection import learning_curve

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), scoring='accuracy'):
    plt.figure(figsize=(10,6))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(scoring)
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, scoring=scoring,
                                                            n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")
    plt.legend(loc="best")
    return plt

train = pd.read_csv('input/train.csv')
test = pd.read_csv('input/test.csv')

for c in train:
    train[c] = pd.Categorical(train[c].values).codes

for c in test:
    test[c] = pd.Categorical(test[c].values).codes

X = train.drop(['SalePrice'], axis=1)
#X = train[['OverallQual', 'GarageArea', 'GarageCars', 'TotalBsmtSF', 'TotRmsAbvGrd', 'FullBath', 'GrLivArea']]
y = train.SalePrice

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

from sklearn import linear_model
lr = linear_model.LinearRegression()
lr.fit(X_train, y_train)
lr.score(X_train, y_train)

#from sklearn import cross_validation
#scores = cross_validation.cross_val_score(lr, X, y)
#scores.mean()

from sklearn.model_selection import cross_val_predict
predicted = cross_val_predict(lr, X, y, cv=10)
fig, ax = plt.subplots()
ax.scatter(y, predicted, color='red')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=4)
ax.set_xlabel('Measured')
ax.set_ylabel('Predicted')
plt.show()

#from sklearn.linear_model import RidgeCV
#clf_ridge = lm.Ridge()
#clf_ridge.fit(X, y, sample_weight=None)
#pred_train = predicted.predict(X)

clf_ridge_cv = lm.RidgeCV()
clf_ridge_cv.fit(X, y, sample_weight=None)
pred_train_cv = clf_ridge_cv.predict(X)
predicted = cross_val_predict(clf_ridge_cv, X, y, cv=10)

print(predicted.mean())
print(pred_train_cv.mean())

plt = plot_learning_curve(clf_ridge_cv, 'ridge_model', X, y, train_sizes=[200,400,600,800,100,1200,1300], cv=10, scoring='neg_mean_squared_error')
plt.show()
#pred_test= clf_ridge.predict(test)

#plt.figure(figsize=(10,6))
#plt.title("train vs test")
#plt.xlabel("data")
#plt.ylabel("estimation")
#plt.plot(pred_train, 'o-', color="r", label="Train predict")
#plt.plot(pred_train_cv, 'o-', color="g", label="Test predict")
#plt.show()