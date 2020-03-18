import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from automobileRegression.dataConverter import replace_cat, replace_dum, delete_missing
from automobileRegression.validation import polynomial_kcrossVal_dummy, polynomial_kcrossVal, polynomial_kcrossVal_cat, multi_ind_gridCV
from automobileRegression.analisis import learning_curves
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

df = pd.read_csv("../data/imports-85.csv")
df_copy = df.copy()

#determining best independent variables with k-cross validation method (uncomment to see the ratio and best parameters)
# print("symbolling /car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.symboling, df_copy.price, color='black')
# plt.xlabel('symboling')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['symboling'], ['price'], 10)
# print("")
#
# print("normalized losses /car price k-cross validation(neg_mean_squared_error) ratio:")
# df2 = delete_missing(df_copy, ['normalized_losses', 'price'])
# plt.scatter(df2.normalized_losses, df2.price, color='black')
# plt.xlabel('normalized losses')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df2, ['normalized_losses'], ['price'], 10)
# print("")
# del df2
#
# print("car make/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.make, df_copy.price, color='black')
# plt.xlabel('car make')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_dummy(df_copy, ['make'], ['price'], 10)
# print("")
#
# print("fuel-type/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.fuel_type, df_copy.price, color='black')
# plt.xlabel('fuel type')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['fuel_type'], ['price'], {'fuel_type': {'gas': 0, 'diesel': 1}}, 10)
# print("")
#
# print("car horsepower/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.horsepower, df_copy.price, color='black')
# plt.xlabel('horsepower')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['horsepower'], ['price'], 10)
# print("")
#
# print("aspiration/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.aspiration, df_copy.price, color='black')
# plt.xlabel('aspiration')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['aspiration'], ['price'], {'aspiration': {'std': 0, 'turbo': 1}}, 10)
# print("")
#
# print("number of doors/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.num_of_doors, df_copy.price, color='black')
# plt.xlabel('number of doors')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['num_of_doors'], ['price'], {'num_of_doors': {'two': 0, 'four': 1}}, 10)
# print("")
#
# print("body style/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.body_style, df_copy.price, color='black')
# plt.xlabel('body style')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['body_style'], ['price'], {'body_style': {'hatchback': 0,
#                                                                              'sedan': 1,
#                                                                              'wagon': 2,
#                                                                              'convertible': 3,
#                                                                              'hardtop': 4}}, 10)
# print("")
#
# print("drive wheels/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.drive_wheels, df_copy.price, color='black')
# plt.xlabel('drive wheels')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['drive_wheels'], ['price'], {'drive_wheels': {'fwd': 0, 'rwd': 1, '4wd': 2}}, 10)
# print("")
#
# print("engine location/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.engine_location, df_copy.price, color='black')
# plt.xlabel('engine location')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['engine_location'], ['price'], {'engine_location': {'front': 0, 'rear': 1}}, 10)
# print("")
#
#
# print("wheel base/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.wheel_base, df_copy.price, color='black')
# plt.xlabel('wheel base')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['wheel_base'], ['price'], 10)
# print("")
#
# print("length/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.length, df_copy.price, color='black')
# plt.xlabel('length')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['length'], ['price'], 10)
# print("")
#
# print("width/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.width, df_copy.price, color='black')
# plt.xlabel('width')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['width'], ['price'], 10)
# print("")
#
#
# print("height/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.height, df_copy.price, color='black')
# plt.xlabel('height')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['height'], ['price'], 10)
# print("")
#
# print("curb weight/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.curb_weight, df_copy.price, color='black')
# plt.xlabel('curb weight')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['curb_weight'], ['price'], 10)
# print("")
#
# print("engine type/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.engine_type, df_copy.price, color='black')
# plt.xlabel('engine type')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_dummy(df_copy, ['engine_type'], ['price'], 10)
# print("")
#
# print("number of cylinders/ car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.num_of_cylinders, df_copy.price, color='black')
# plt.xlabel('number of cylinders')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_cat(df_copy, ['num_of_cylinders'], ['price'], {'num_of_cylinders': {'two': 0,
#                                                                                          'three': 1,
#                                                                                          'four': 2,
#                                                                                          'five': 3,
#                                                                                          'six': 4,
#                                                                                          'eight': 5,
#                                                                                          'twelve': 6}}, 10)
# print("")
#
# print("engine size/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.engine_size, df_copy.price, color='black')
# plt.xlabel('engine size')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['engine_size'], ['price'], 10)
# print("")
#
# print("fuel system/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.fuel_system, df_copy.price, color='black')
# plt.xlabel('fuel system')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal_dummy(df_copy, ['fuel_system'], ['price'], 10)
# print("")
#
# print("bore/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.bore, df_copy.price, color='black')
# plt.xlabel('bore')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['bore'], ['price'], 10)
# print("")
#
# print("stroke/car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.stroke, df_copy.price, color='black')
# plt.xlabel('stroke')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['stroke'], ['price'], 10)
# print("")
#
# print("compression ratio /car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.compression_ratio, df_copy.price, color='black')
# plt.xlabel('compression ratio')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['compression_ratio'], ['price'], 10)
# print("")
#
# print("peak rpm /car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.peak_rpm, df_copy.price, color='black')
# plt.xlabel('peak rpm')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['peak_rpm'], ['price'], 10)
# print("")
#
# print("city mpg /car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.city_mpg, df_copy.price, color='black')
# plt.xlabel('city_mpg')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['city_mpg'], ['price'], 10)
# print("")
#
# print("highway mpg /car price k-cross validation(neg_mean_squared_error) ratio:")
# plt.scatter(df_copy.highway_mpg, df_copy.price, color='black')
# plt.xlabel('highway mpg')
# plt.ylabel('price')
# plt.yticks([])
# plt.xticks([])
# plt.show()
# polynomial_kcrossVal(df_copy, ['highway_mpg'], ['price'], 10)
# print("")

param_tab = ['horsepower', 'highway_mpg', 'engine_size', 'fuel_system', 'price']
cdf = df_copy[param_tab]
cdf = delete_missing(cdf, param_tab)
cdf = replace_dum(cdf, ['fuel_system'])
ind_var = list(cdf.columns.values)
ind_var.remove('price')
X = cdf[ind_var]
y = cdf[['price']]
multi_ind_gridCV(X, y)

from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
lm = linear_model.LinearRegression(fit_intercept=True, normalize=False)
poly = PolynomialFeatures(degree=2)
polyX = poly.fit_transform(X_train)
train_y_ = lm.fit(polyX, y_train)
polyXTest = poly.fit_transform(X_test)
y_pred = lm.predict(polyXTest)
print("MSE: " + str(mean_squared_error(y_test, y_pred)))



learning_curves(X, y, 2, True, False)










