import pandas as pd

main_file_path = 'dataset/train.csv'
data = pd.read_csv(main_file_path)

# print(data.describe())
s = data.SalePrice
# print(s.describe())
b = data[['OverallQual','LotArea','GrLivArea',]]
# print( b.head())
y = data.SalePrice
# print( y.head())
X = b
# print (x.head())
from sklearn.model_selection import train_test_split
train_X, val_X, train_y, val_y = train_test_split(X, y,random_state = 0)
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

def get_mae(max_leaf,pred_train,pred_val,targ_train,targ_val):
    model = RandomForestRegressor(max_leaf_nodes=max_leaf,random_state=0)
    model.fit(pred_train,pred_val)
    preds_val = model.predict(targ_train)
    mae = mean_absolute_error(targ_val,preds_val)
    return(mae)
for a in [5,50,70,5000]:
    my_mae = get_mae(a,train_X,train_y,val_X,val_y)

    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" %(a, my_mae))
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(max_leaf_nodes= 70)
model.fit(X , y)
melb_preds = model.predict(val_X)
print(mean_absolute_error(val_y , melb_preds))
test = pd.read_csv('dataset/test.csv')
test_X =test[['OverallQual','LotArea','GrLivArea']]
pred = model.predict(test_X)
print(pred)
