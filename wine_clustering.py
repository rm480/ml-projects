#in progress
import pandas as pd
df = pd.read_csv("winequality-red.csv")
temp=df.describe()
df.isnull().values.any()
df.loc[df['total sulfur dioxide']>200, 'total sulfur dioxide'] = 180
df['quality']=[1 if df['quality'][row]>6 else 0 for row in df['quality']]
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
x_train, x_test, y_train, y_test = train_test_split(df.iloc[:,0:11], df['quality'], train_size=0.8, random_state=123)
scaler = StandardScaler(copy=False)
scaler.fit(x_train)
scaler.transform(x_train)
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
from scipy.stats import uniform
paramsCV = dict(
            reg_lambda=uniform(loc=0, scale=10), min_child_weight=uniform(loc=0.1, scale=1),
                gamma=uniform(loc=0, scale=0.3))
xgbCV = RandomizedSearchCV(xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.35, min_child_weight=0.5,
        max_depth=20, gamma=0.2, reg_alpha=6.5), paramsCV,
                     scoring='accuracy', cv=10, n_jobs=-1)
xgbCV.fit(x_train, y_train)
xgbCV.best_params_
modelxgb = xgb.XGBClassifier(objective='binary:logistic', learning_rate=0.35, min_child_weight=0.5,
                  max_depth=20, gamma=0.2, reg_alpha=6.5)
modelxgb.fit(x_train,y_train)
modelxgb.feature_importances_


#clustering
from sklearn.decomposition import PCA
pca = PCA(n_components = 3, copy=False)
scaler = StandardScaler(copy=False)
scaled = pd.DataFrame(scaler.fit_transform(df.iloc[:,0:11]), columns=df.iloc[:,0:11].columns)
reduced = pca.fit_transform(scaled)
scaled = pd.concat([scaled, df['quality']], axis=1)

from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
tsne = pd.DataFrame(TSNE(n_components=3).fit_transform(scaled), columns=['first', 'second', 'third'])
tsne = pd.concat([tsne, df['quality']], axis=1)
fig = px.scatter_3d(tsne, x='first', y='second', z='third', color='quality')
fig.show()


from pycaret.classification import *
setup = setup(df, target = 'quality', train_size=0.8, normalize=True)
compare_models()
model = create_model('xgboost')
tuned_model = tune_model('xgboost', n_iter=10)
plot_model(tuned_model, 'learning')
predicted = predict_model(tuned_model)
