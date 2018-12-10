import pandas as pd
import numpy as np
train=pd.read_csv("train.csv")
test=pd.read_csv("test.csv")
train_labels=train['SalePrice'].copy()
train=train.drop(columns='SalePrice')
train_labels=train_labels.apply(np.log)
#transformation
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=np.nan, strategy='median')
train['LotFrontage']=imp.fit_transform(train[['LotFrontage']]).ravel()
train['GarageYrBlt']=imp.fit_transform(train[['GarageYrBlt']]).ravel()
put_nas = ['MasVnrType','GarageType','GarageQual','GarageFinish','GarageCond','Electrical','Alley','BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2','FireplaceQu','PoolQC','Fence','MiscFeature']
train[put_nas] = train[put_nas].fillna('No item')
train["MasVnrArea"].fillna(0, inplace=True)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
col_string = [train.columns[i] for i in range(len(train.columns)) if isinstance(train.iloc[0,i], str)]
cat_attribs = train[col_string]
cat_pipeline = Pipeline([
    ('label_binarizer', OneHotEncoder(handle_unknown='ignore', sparse=False))
])
num_pipeline = Pipeline([
    ('std_scaler', StandardScaler())
])
temp1 = num_pipeline.fit_transform(train.drop(col_string, axis=1))
temp2 = cat_pipeline.fit_transform(train[col_string])
train_clean = pd.concat([pd.DataFrame(temp1), pd.DataFrame(temp2)], axis=1, ignore_index=True)

#gradient descent
from sklearn.linear_model import SGDRegressor
sgd_reg = SGDRegressor(max_iter=1000, penalty=None, learning_rate='invscaling', eta0=0.1)
sgd_reg.fit(train_clean, train_labels.ravel())

from sklearn.metrics import mean_squared_error
#random forest
from sklearn.ensemble import RandomForestRegressor
rnd_reg = RandomForestRegressor(n_estimators=50, max_features=50, n_jobs=-1)
rnd_reg.fit(train_clean, train_labels)
labels_predict = rnd_reg.predict(train_clean)
print(np.sqrt(mean_squared_error(train_labels, labels_predict)))

#grid search
from sklearn.model_selection import GridSearchCV
param_grid = [
    {'max_features':[5,10,25,40,50]}
]
grid_search = GridSearchCV(RandomForestRegressor(n_estimators = 50), param_grid, cv = 10,
                           scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(train_clean, train_labels)
grid_predict = grid_search.predict(train_clean)
print(np.sqrt(mean_squared_error(train_labels, grid_predict)))

#cross-validation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(rnd_reg, train_clean, train_labels,
                         scoring='neg_mean_squared_error', cv=10, n_jobs=-1)
print(np.sqrt(-scores))
#simple linear regression
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(train_clean, train_labels)
print(np.sqrt(mean_squared_error(train_labels, lin_reg.predict(train_clean))))