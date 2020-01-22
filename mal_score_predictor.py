from jikanpy import Jikan
import pandas as pd
import numpy as np
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
        for anime in range(len(all_seasons[season][year]['anime'])):
            id.append(all_seasons[season][year]['anime'][anime]['mal_id'])
id = pd.DataFrame(id)
id = pd.DataFrame(id[0].unique())
with open('id.data', 'wb') as filehandle:
    pickle.dump(id, filehandle)
with open('id.data', 'rb') as filehandle:
    id = pickle.load(filehandle)
id = id.to_numpy().transpose().flatten()
anime = list()
errors = list()
for i in errors:
    try:
        anime.append(jikan.anime(i))
    except:
        errors.append(i)
        pass
    time.sleep(4)

# data preparation
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MultiLabelBinarizer, OneHotEncoder
with open('anime.data', 'rb') as filehandle:
    anime = pickle.load(filehandle)
anime_pd = pd.DataFrame(anime)[['title', 'type', 'source', 'episodes', 'airing', 'aired', 'duration',
    'rating', 'scored_by', 'members', 'favorites', 'related', 'synopsis', 'producers',
    'licensors', 'studios', 'genres', 'opening_themes', 'ending_themes', 'trailer_url', 'score']]
anime_pd = anime_pd.loc[(anime_pd['airing']==False)&(anime_pd['rating']!='Rx - Hentai')].reset_index(drop=True)
anime_pd['from'] = [anime_pd['aired'][i]['from'] for i in range(len(anime_pd['aired']))]
anime_pd['to'] = [anime_pd['aired'][i]['to'] for i in range(len(anime_pd['aired']))]
anime_pd['from']=anime_pd['from'].str.split('T',expand=True)[0]
anime_pd['to']=anime_pd['to'].str.split('T',expand=True)[0]
list_cols=['producers','licensors','studios','genres']
for feature in list_cols:
    for row in range(len(anime_pd[feature])):
        list_i = 0
        while list_i < len(anime_pd[feature][row]):
            anime_pd[feature][row][list_i]=anime_pd[feature][row][list_i]['name']
            list_i += 1
anime_pd['duration'] = anime_pd['duration'].str.split(' per', expand=True)[0]
anime_pd['duration'] = anime_pd['duration'].str.split(' min', expand=True)[0]
for row in range(len(anime_pd['duration'])):
    if 'hr ' in anime_pd['duration'][row]:
        anime_pd['duration'][row] = str(int(anime_pd['duration'][row].split(' ')[0])*60\
                                    + int(anime_pd['duration'][row].split(' ')[2]))
    if 'hr' in anime_pd['duration'][row]:
        anime_pd['duration'][row] = str(int(anime_pd['duration'][row].split(' ')[0])*60)
    if 'sec' in anime_pd['duration'][row]:
        anime_pd['duration'][row] = str(int(anime_pd['duration'][row].split(' ')[0])/60)
    if 'Unknown' in anime_pd['duration'][row]:
        anime_pd['duration'][row] = '25'
cols_to_num = ['duration', 'scored_by', 'members', 'favorites', 'from']
for col in cols_to_num:
    anime_pd[col] = anime_pd[col].astype(float)
anime_pd = pd.concat([anime_pd, anime_pd['related'].apply(pd.Series)], axis=1)
related=list(anime_pd.loc[:,'Summary':'Spin-off'].columns)
for feature in related:
    anime_pd[feature] = pd.DataFrame(np.where(anime_pd[feature].isna(),0,anime_pd[feature]))
    for row in range(len(anime_pd[feature])):
        if anime_pd[feature][row] != 0:
            anime_pd[feature][row] = len(anime_pd[feature][row])
anime_pd['season'] = (pd.to_numeric(anime_pd['from'].str.split('-', 2, expand=True)[1])%12+3)//3
anime_pd['from'] = pd.to_timedelta(pd.to_datetime(anime_pd['from']), unit='d').dt.days
anime_pd['to'] = pd.to_timedelta(pd.to_datetime(anime_pd['to']), unit='d').dt.days
anime_pd['to'].fillna(anime_pd['from'], inplace=True)
anime_pd['trailer_url'] = pd.DataFrame(np.where(anime_pd['trailer_url'].isna(),0,1))
anime_pd['opening_themes'] = pd.DataFrame([len(anime_pd.iloc[row,17])
                                           for row in range(len(anime_pd['opening_themes']))])
anime_pd['ending_themes'] = pd.DataFrame([len(anime_pd.iloc[row,18])
                                          for row in range(len(anime_pd['ending_themes']))])
cols_to_multihot = ['producers','licensors','studios','genres']
cols_to_onehot = ['type','source','rating','season']
one_hot = OneHotEncoder()
mlb = MultiLabelBinarizer()
anime_pd = pd.concat([anime_pd, pd.DataFrame(mlb.fit_transform(anime_pd[feature]))], axis=1)



test=one_hot.fit_transform(anime_pd[['type']])
anime_pd['rating'].unique()
anime_pd['to'].isna().sum()
