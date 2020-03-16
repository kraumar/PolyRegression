from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from automobileRegression.dataConverter import replace_cat, replace_dum, delete_missing
from sklearn import linear_model
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline

lm = linear_model.LinearRegression()

def PolynomialRegression(degree=2, **kwargs):
    return make_pipeline(PolynomialFeatures(degree), LinearRegression(**kwargs))


def polynomial_kcrossVal(df_copy, ind_var, dep_var, degree, k):
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


def polynomial_kcrossVal_dummy(df_copy, ind_var, dep_var, degree, k):
    crossvalidation = KFold(n_splits=k)
    cdf = df_copy[ind_var + dep_var]
    cdf = delete_missing(cdf, ind_var + dep_var)
    cdf = replace_dum(cdf, ind_var)
    poly = PolynomialFeatures(degree=degree)
    cdf_list = list(cdf.columns.values)
    cdf_list.remove(dep_var[0])
    polyX = poly.fit_transform(cdf[cdf_list])
    model = lm.fit(polyX, cdf[dep_var])
    scores = cross_val_score(model, polyX, cdf[dep_var], cv=crossvalidation)
    print(np.mean(np.abs(scores)))

    print(cdf_list)
    print(cdf)

    param_grid = {'polynomialfeatures__degree': np.arange(20),
                  'linearregression__fit_intercept': [True, False],
                  'linearregression__normalize': [True, False]}
    grid = GridSearchCV(PolynomialRegression(), param_grid, cv=k)
    grid.fit(cdf['make_alfa-romero', 'make_audi'], cdf[dep_var])




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







