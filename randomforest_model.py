import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
In [2]:
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import Imputer
In [3]:
data = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
# print(len(test.columns))
X = data.drop(['SalePrice'], axis=1)
# print(X.describe())
X = pd.get_dummies(X)
# print(X.describe())
print(X.columns)
test = pd.get_dummies(test)
y = data.SalePrice
# X = X.drop(['HouseStyle_2.5Fin'], axis=1)
c = list(X.columns)
d = list(test.columns)
# f = list(set(c).intersection(d))
e = list(set(c).difference(d))
print(len(e))
for i in e:
#     print(i)  
    X = X.drop(i, axis=1)

#vprint(test.describe())
print(X.columns)
# print(X.describe())
X_train, X_test, y_train, y_test = train_test_split(X, y,random_state = 0)

my_imputer = Imputer()
imputed_X_train = my_imputer.fit_transform(X)
imputed_X_test = my_imputer.transform(test)
#print("Mean Absolute Error from Imputation:")
def get_mae(max_leaf,pred_train,pred_val,targ_train,targ_val):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf,random_state=0)
    model.fit(pred_train,pred_val)
    preds_val = model.predict(targ_train)
    print (preds_val)
#     mae = mean_absolute_error(targ_val,preds_val)
#     return(mae)
print(get_mae(700,imputed_X_train,  y,imputed_X_test, y))    

model = RandomForestRegressor(max_leaf_nodes= 70)
my_imputer = Imputer()
print(X.columns)
print(test.columns)

imputed_X_train = my_imputer.fit_transform(X)
imputed_X_test = my_imputer.transform(test)

model.fit(imputed_X_train, y)
pred = model.predict(imputed_X_test)

print(pred)
