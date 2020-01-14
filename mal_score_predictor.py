from jikanpy import Jikan
import pandas as pd
import time
import pickle
jikan = Jikan()
#anime = jikan.anime(39597)
#list.extend
winter = list()
spring = list()
summer = list()
fall = list()
years = range(1980,2019)
for Y in years:
    time.sleep(4)
    winter.append(jikan.season(year=2008, season='winter'))
with open('winter.data', 'wb') as filehandle:
    pickle.dump(winter, filehandle)
with open('winter.data', 'rb') as filehandle:
    winter = pickle.load(filehandle)
#winter[0]['anime'][0]['mal_id']
for Y in years:
    try:
        spring.append(jikan.season(year=2006, season='spring'))
    except:
        pass
    time.sleep(4)
with open('spring.data', 'wb') as filehandle:
    pickle.dump(spring, filehandle)
for Y in years:
    time.sleep(4)
    try:
        summer.append(jikan.season(year=1993, season='summer'))
    except:
        pass
with open('summer.data', 'wb') as filehandle:
    pickle.dump(summer, filehandle)
for Y in years:
    time.sleep(4)
    try:
        fall.append(jikan.season(year=2005, season='fall'))
    except:
        pass
with open('fall.data', 'wb') as filehandle:
    pickle.dump(fall, filehandle)


all_seasons = [winter, spring, summer, fall]
with open('all_seasons.data', 'wb') as filehandle:
    pickle.dump(all_seasons, filehandle)
with open('all_seasons.data', 'rb') as filehandle:
    all_seasons = pickle.load(filehandle)
id = list()
for season in range(3):
    for year in range(39):
        for anime in range(len(all_seasons[season][year]['anime'])):
            id.append(all_seasons[season][year]['anime'][anime]['mal_id'])
id = pd.DataFrame(id)
id = pd.DataFrame(id[0].unique())
with open('id.data', 'wb') as filehandle:
    pickle.dump(id, filehandle)
