#Prediction of anime score from MyAnimeList database

from jikanpy import Jikan
import numpy as np
import time
jikan = Jikan()
anime = [None]*2
anime = jikan.anime(39597) #1-10507 + 30000 smth take ids from seasons below

years = range(2017,2019)
winter = [jikan.season(year=Y, season='winter') for Y in years]
spring = [jikan.season(year=Y, season='spring') for Y in years]
summer = [jikan.season(year=Y, season='summer') for Y in years]
fall = [jikan.season(year=Y, season='fall') for Y in years]
db = pd.DataFrame([winter[0]['anime'], spring[0]['anime'], summer[0]['anime'], fall[0]['anime']])
