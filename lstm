import pandas as pd
import numpy as np
rmrf=pd.read_csv("rmrf.csv")
train=rmrf.iloc[0:3999,1:2]
test=rmrf.iloc[3999:4847,1:2]
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_train=scaler.fit_transform(train)

from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
model = Sequential()
model.add(LSTM(units=50, activation='relu',
               return_sequences=True, batch_input_shape=(200,1), stateful=True))
