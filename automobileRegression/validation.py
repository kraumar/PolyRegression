from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from automobileRegression.dataConverter import replace_cat, replace_dum, delete_missing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
import numpy as np


lm = linear_model.LinearRegression()

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def polynomial_kcrossVal(df_copy, ind_var, dep_var, k):
    cdf = df_copy[ind_var + dep_var]
    cdf = delete_missing(cdf, ind_var + dep_var)
    cdf_list = list(cdf.columns.values)
    cdf_list.remove(dep_var[0])
    param_grid = {'polynomialfeatures__degree': np.arange(20),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=k)
    grid.fit(cdf[ind_var], cdf[dep_var])
    print(np.abs(grid.best_score_))
    print(grid.best_params_)


def polynomial_kcrossVal_dummy(df_copy, ind_var, dep_var, k):
    crossvalidation = KFold(n_splits=k)
    cdf = df_copy[ind_var + dep_var]
    cdf = delete_missing(cdf, ind_var + dep_var)
    cdf = replace_dum(cdf, ind_var)
    cdf_list = list(cdf.columns.values)
    cdf_list.remove(dep_var[0])
    poly = PolynomialFeatures(degree=1)
    polyX = poly.fit_transform(cdf[cdf_list])
    model = lm.fit(polyX, cdf[dep_var])
    scores = cross_val_score(model, polyX, cdf[dep_var], cv=crossvalidation)
    print(np.mean(np.abs(scores)))





def polynomial_kcrossVal_cat(df_copy, ind_var, dep_var, mapX, k):
    cdf = df_copy[ind_var + dep_var]
    cdf = delete_missing(cdf, ind_var + dep_var)
    cdf = replace_cat(cdf, mapX)
    cdf_list = list(cdf.columns.values)
    cdf_list.remove(dep_var[0])
    param_grid = {'polynomialfeatures__degree': np.arange(20),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=k)
    grid.fit(cdf[ind_var], cdf[dep_var])
    print(np.abs(grid.best_score_))
    print(grid.best_params_)

def multi_ind_gridCV(X, y):
    bool_tab = [True, False]
    best_degree = -1
    best_score = -1
    best_param1 = False
    best_param2 = False
    for i in range(1, 6):
        for x in bool_tab:
            for j in bool_tab:
                poly = PolynomialFeatures(degree=i)
                lm = linear_model.LinearRegression(fit_intercept=x, normalize=j)
                polyX = poly.fit_transform(X)
                model = lm.fit(polyX, y)
                scores = cross_val_score(model, polyX, y, cv=10, scoring='r2')
                score = np.mean(np.abs(scores))
                if score > best_score and score <= 1:
                    best_score = score
                    best_degree = i
                    best_param1 = x
                    best_param2 = j
    print("{degree: " + str(best_degree) + ", fit_intercept: " + str(best_param1) + ", normalize: " +
          str(best_param2) + "}")
    print("r2 score: " + str(best_score))




