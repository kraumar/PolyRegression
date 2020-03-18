from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_score
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
import numpy as np


def learning_curves(X, y, degree, fit_intercept, normalize):
    tab = np.arange(0.9, 0.1, -0.08)
    train_scores_tab = np.zeros(tab.size-1)
    crossval_scores = np.zeros(tab.size-1)
    x = 0
    poly = PolynomialFeatures(degree)
    for i in range(1,tab.size):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=tab[i], random_state=42)
        polyX = poly.fit_transform(X_train)
        polyXval = poly.fit_transform(X_val)
        lm = linear_model.LinearRegression(fit_intercept=fit_intercept, normalize=normalize)
        lm.fit(polyX, y_train)
        train_scores = cross_validate(lm, polyX, y_train, cv=10, scoring='r2', return_train_score=True)
        train_scores_tab[x] = train_scores['train_score'].mean()
        crossval_scores[x] = np.abs(cross_val_score(lm, polyX, y_train, cv=10, scoring='r2').mean())
        x = x+1

    print(train_scores_tab)
    print(crossval_scores)