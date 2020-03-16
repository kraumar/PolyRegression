import numpy as np
import sklearn
import matplotlib.pyplot as plt
import pandas as pd
from automobileRegression.dataConverter import replace_cat, replace_dum, delete_missing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from automobileRegression.validation import polynomial_kcrossVal_dummy, polynomial_kcrossVal, polynomial_kcrossVal_cat

df = pd.read_csv("../data/imports-85.csv")
df_copy = df.copy()

#determining best independent variables with k-cross validation method
print("symbolling /car price k-cross validation(neg_mean_squared_error) ratio:")
plt.scatter(df_copy.symboling, df_copy.price, color='black')
plt.xlabel('symboling')
plt.ylabel('price')
plt.yticks([])
plt.xticks([])
plt.show()
polynomial_kcrossVal(df_copy, ['symboling'], ['price'], 1, 4)
print("")

print("normalized losses /car price k-cross validation(neg_mean_squared_error) ratio:")
df2 = delete_missing(df_copy, ['normalized_losses', 'price'])
plt.scatter(df2.normalized_losses, df2.price, color='black')
plt.xlabel('normalized losses')
plt.ylabel('price')
plt.yticks([])
plt.xticks([])
plt.show()
polynomial_kcrossVal(df2, ['normalized_losses'], ['price'], 2, 4)
print("")
del df2

print("car make/car price k-cross validation(neg_mean_squared_error) ratio:")
plt.scatter(df_copy.make, df_copy.price, color='black')
plt.xlabel('car make')
plt.ylabel('price')
plt.yticks([])
plt.xticks([])
plt.show()
polynomial_kcrossVal_dummy(df_copy, ['make'], ['price'], 1, 4)
print("")

print("fuel-type/ car price k-cross validation(neg_mean_squared_error) ratio:")
plt.scatter(df_copy.fuel_type, df_copy.price, color='black')
plt.xlabel('fuel type')
plt.ylabel('price')
plt.yticks([])
plt.xticks([])
plt.show()
polynomial_kcrossVal_cat(df_copy, ['fuel_type'], ['price'], {'fuel_type': {'gas': 0, 'diesel': 1}}, 4)
print("")

print("car horsepower/car price k-cross validation(neg_mean_squared_error) ratio:")
plt.scatter(df_copy.horsepower, df_copy.price, color='black')
plt.xlabel('horsepower')
plt.ylabel('price')
plt.yticks([])
plt.xticks([])
plt.show()
polynomial_kcrossVal(df_copy, ['horsepower'], ['price'], 2, 4)
print("")

