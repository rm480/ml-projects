from jikanpy import Jikan
import pandas as pd
import time
import pickle
jikan = Jikan()
winter = list()
spring = list()
summer = list()
fall = list()
years = range(1980, 2019)
for Y in years:
    time.sleep(4)
    winter.append(jikan.season(year=2008, season='winter'))
for Y in years:
    try:
        spring.append(jikan.season(year=2006, season='spring'))
    except:
        pass
    time.sleep(4)
for Y in years:
    time.sleep(4)
    try:
        summer.append(jikan.season(year=1993, season='summer'))
    except:
        pass
for Y in years:
    time.sleep(4)
    try:
        fall.append(jikan.season(year=2005, season='fall'))
    except:
        pass
with open('fall.data', 'wb') as filehandle:
    pickle.dump(fall, filehandle)

all_seasons = [winter, spring, summer, fall]
id = list()
for season in range(3):
    for year in range(39):
        for title in range(len(all_seasons[season][year]['anime'])):
            id.append(all_seasons[season][year]['anime'][title]['mal_id']) 
id = pd.DataFrame(id)
id = pd.DataFrame(id[0].unique())
with open('id.data', 'wb') as filehandle:
    pickle.dump(id, filehandle)
with open('id.data', 'rb') as filehandle:
    id = pickle.load(filehandle)
id = id.to_numpy().transpose().flatten()
title = list()
errors = list()
for i in errors:
    try:
        title.append(jikan.title(i))
    except:
        errors.append(i)
        pass
    time.sleep(4)

# data preparation
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
with open('title.data', 'rb') as filehandle:
    title = pickle.load(filehandle)
title_pd = pd.DataFrame(title)[['title', 'type', 'source', 'episodes', 'airing', 'aired', 'duration',
                                'rating', 'scored_by', 'members', 'favorites', 'related', 'synopsis', 'producers',
                                'studios', 'genres', 'opening_themes', 'ending_themes', 'trailer_url', 'score']]
title_pd = title_pd.loc[(title_pd['airing']==False)&(title_pd['rating']!='Rx - Hentai')].reset_index(drop=True)
title_pd = pd.concat([pd.DataFrame(title_pd.index,columns=['index']),title_pd], axis=1, sort=False)
title_pd['from'] = [title_pd['aired'][i]['from'] for i in range(len(title_pd['aired']))]
title_pd['to'] = [title_pd['aired'][i]['to'] for i in range(len(title_pd['aired']))]
title_pd['from']=title_pd['from'].str.split('T',expand=True)[0]
title_pd['to']=title_pd['to'].str.split('T',expand=True)[0]
list_cols=['producers','studios','genres']
for feature in list_cols:
    for row in range(len(title_pd[feature])):
        list_i = 0
        while list_i < len(title_pd[feature][row]):
            title_pd[feature][row][list_i]=title_pd[feature][row][list_i]['name']
            list_i += 1
title_pd['duration'] = title_pd['duration'].str.split(' per', expand=True)[0]
title_pd['duration'] = title_pd['duration'].str.split(' min', expand=True)[0]
for row in range(len(title_pd['duration'])):
    if 'hr ' in title_pd['duration'][row]:
        title_pd['duration'][row] = str(int(title_pd['duration'][row].split(' ')[0])*60 \
                                        + int(title_pd['duration'][row].split(' ')[2]))
    if 'hr' in title_pd['duration'][row]:
        title_pd['duration'][row] = str(int(title_pd['duration'][row].split(' ')[0])*60)
    if 'sec' in title_pd['duration'][row]:
        title_pd['duration'][row] = str(int(title_pd['duration'][row].split(' ')[0])/60)
    if 'Unknown' in title_pd['duration'][row]:
        title_pd['duration'][row] = '25'
cols_to_num = ['duration', 'scored_by', 'members', 'favorites']
for col in cols_to_num:
    title_pd[col] = title_pd[col].astype(float)
title_pd = pd.concat([title_pd, title_pd['related'].apply(pd.Series)], axis=1)
related=list(title_pd.loc[:,'Summary':'Spin-off'].columns)
for feature in related:
    title_pd[feature] = pd.DataFrame(np.where(title_pd[feature].isna(),0,title_pd[feature]))
    for row in range(len(title_pd[feature])):
        if title_pd[feature][row] != 0:
            title_pd[feature][row] = len(title_pd[feature][row])
title_pd['season'] = (pd.to_numeric(title_pd['from'].str.split('-', 2, expand=True)[1])%12+3)//3
for row in range(len(title_pd['season'])):
    if title_pd['season'][row]==1:
        title_pd['season'][row]='winter'
    elif title_pd['season'][row]==2:
        title_pd['season'][row]='spring'
    elif title_pd['season'][row]==3:
        title_pd['season'][row]='summer'
    elif title_pd['season'][row]==4:
        title_pd['season'][row]='fall'
title_pd['from'] = pd.to_timedelta(pd.to_datetime(title_pd['from']), unit='d').dt.days
title_pd['to'] = pd.to_timedelta(pd.to_datetime(title_pd['to']), unit='d').dt.days
title_pd['to'].fillna(title_pd['from'], inplace=True)
title_pd['trailer_url'] = pd.DataFrame(np.where(title_pd['trailer_url'].isna(),0,1))
title_pd['opening_themes'] = pd.DataFrame([len(title_pd.iloc[row,17])
                                           for row in range(len(title_pd['opening_themes']))])
title_pd['ending_themes'] = pd.DataFrame([len(title_pd.iloc[row,18])
                                          for row in range(len(title_pd['ending_themes']))])
title_pd['from'] = title_pd['from'].astype(float)
title_pd['favorites']=title_pd['favorites']/title_pd['members']
cols_to_multihot = ['producers','studios','genres']
temp = [['episodes','duration','scored_by','members','favorites'],
        list(title_pd.loc[:,'opening_themes':'Spin-off'].columns)]
cols_to_num = [item for elem in temp for item in elem]
cols_to_onehot = ['type','source','rating','season']
one_hot = OneHotEncoder(sparse=False)
mlb1 = MultiLabelBinarizer(sparse_output=False)
mlb2 = MultiLabelBinarizer(sparse_output=False)
mlb3 = MultiLabelBinarizer(sparse_output=False)
scaler = StandardScaler()
mlb_pd = pd.concat([pd.DataFrame(title_pd['index']),
                    pd.DataFrame(mlb1.fit_transform(title_pd['producers']),columns=mlb1.classes_+'_p'),
                    pd.DataFrame(mlb2.fit_transform(title_pd['studios']),columns=mlb2.classes_),
                    pd.DataFrame(mlb3.fit_transform(title_pd['genres']),columns=mlb3.classes_+'_g')],
                   axis=1, sort=False)
train, test = train_test_split(title_pd, test_size=0.2, random_state=228)
#test is transformed implicitly to avoid copying the code below by setting train=test
train = train.sort_values(by=['index'])
mlb_pd = mlb_pd[mlb_pd['index'].isin(train['index'])]
trans_train = pd.concat([pd.DataFrame(train['index'],columns=['index']),
                         pd.DataFrame(scaler.fit_transform(train[cols_to_num]),columns=cols_to_num,index=train['index']),
                         pd.DataFrame(one_hot.fit_transform(train[cols_to_onehot]),
                                      columns=[item for elem in one_hot.categories_ for item in elem],index=train['index']),
                         mlb_pd.drop('index',axis=1)], axis=1, sort=False)
cols=trans_train.loc[:,'12 Diary Holders_p':'ufotable'].columns.values
cond=trans_train[cols].sum(axis=0)<15
trans_train.drop(columns=cond[cond==True].index.values, inplace=True)
trans_train['Manga'] = trans_train['Manga']+trans_train['Digital manga']+trans_train['Web manga']
trans_train['Shoujo Ai_g'] = trans_train['Shoujo Ai_g']+trans_train['Yuri_g']
trans_train = trans_train.loc[:,~trans_train.columns.duplicated()]    #Other
trans_train.drop(columns=['Digital manga','Radio','Web manga','Yuri_g'], inplace=True)
with open('train.data', 'wb') as filehandle:
    pickle.dump(trans_train, filehandle)

#model xgboost
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
with open('train.data', 'rb') as filehandle:
    train = pickle.load(filehandle)
with open('test.data', 'rb') as filehandle:
    test = pickle.load(filehandle)
test['Music']=0
test = test[train.columns]
y_train = train['score']
y_test = test['score']
x_train = train.drop(columns=['score','index'])
x_test = test.drop(columns=['score','index'])
paramsCV=[{'learning_rate':[0.1,0.2], 'gamma': [1], 'min_child_weight': [5],
           'max_depth':[6], 'reg_alpha':[5]}]
xgbCV = GridSearchCV(xgb.XGBRegressor(objective='reg:squarederror'), paramsCV,
                     scoring='neg_root_mean_squared_error', cv=5, n_jobs=-1)
xgbCV.fit(x_train, y_train)
xgbCV.best_params_
#consider early stopping
modelxgb=xgb.XGBRegressor(objective='reg:squarederror',learning_rate=0.2,gamma=1,min_child_weight=5,
                          max_depth=6,reg_alpha=5)
modelxgb.fit(x_train,y_train)
pred = pd.DataFrame(modelxgb.predict(x_test))
np.sqrt(mean_squared_error(y_test, pred))
xgb.plot_importance(modelxgb)

#text analysis
import pandas as pd
import numpy as np
import pickle
import string
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
stop_words = stopwords.words('english')
porter = PorterStemmer()
count = CountVectorizer(ngram_range=(1,2), analyzer='word', lowercase=False, min_df=5)
with open('title.data', 'rb') as filehandle:
    title = pickle.load(filehandle)
title_text = pd.DataFrame(title)[['title', 'airing', 'rating', 'synopsis', 'score']]
title_text = title_text.loc[(title_text['airing']==False)&
                            (title_text['rating']!='Rx - Hentai')].reset_index(drop=True)
title_text = pd.concat([pd.DataFrame(title_text.index,columns=['index']),
                        title_text['synopsis']], axis=1, sort=False)
title_text['synopsis'].fillna('None', inplace=True)
for row in range(len(title_text['synopsis'])):
    title_text['synopsis'][row]=title_text['synopsis'][row].strip().lower() \
        .translate(str.maketrans('', '', string.punctuation))
    title_text['synopsis'][row]=word_tokenize(title_text['synopsis'][row])
    title_text['synopsis'][row]=[porter.stem(word) for word in title_text['synopsis'][row]
                                 if word not in stop_words]
temp = [" ".join(synopsis) for synopsis in title_text['synopsis'].values]
bag = count.fit_transform(temp)
with open('train.data', 'rb') as filehandle:
    train = pickle.load(filehandle)
with open('test.data', 'rb') as filehandle:
    test = pickle.load(filehandle)
x_train = bag[train['index']]
x_test = bag[test['index']]
y_train = train['score']
y_test = test['score']
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
model = Sequential()
model.add(Dense(100, input_dim=10885, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam',
              loss='mean_squared_logarithmic_error',
              metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=50, epochs=100)
pred = model.predict(x_test)
