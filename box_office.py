import pandas as pd
import numpy as np
import scipy as sp
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

standardized_features = scaler.fit_transform(train[['popularity','revenue','runtime']])
import datawig
df_train, df_test = datawig.utils.random_split(train)
imputer = datawig.SimpleImputer(
    input_columns=['popularity','revenue','runtime'],
    output_column='budget',
    output_path='imputer_model',
    )
imputer.fit(train_df=df_train,num_epochs=100)
imputed = imputer.predict(train)
del df_test, df_train
temp = imputed.copy()
temp['budget']=imputed['budget'].where(imputed['budget']!=0, imputed['budget_imputed'],axis=0)
imputed['budget']=temp['budget']
imputed['budget'].where(imputed['budget']>0, 0, inplace=True)
from fancyimpute import KNN
