import GetOldTweets3 as got
import pandas as pd
tweetCriteria = got.manager.TweetCriteria().setQuerySearch('unicredit') \
    .setSince("2019-12-20").setUntil("2019-12-21")
tweet = got.manager.TweetManager.getTweets(tweetCriteria)
tweet_date = [tweet[i].date for i in range(len(tweet))]
tweet_date_df = pd.DataFrame(tweet_date)
tweet_date_df.to_csv(path_or_buf='C:\\Users\\linai\\Documents\\tweets\\@2012.csv', index=False)

#start here
import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
time = pd.read_csv('diurtime.csv').to_numpy().transpose().flatten().astype(float)

def kernel(x,y,b):
    return np.exp(-b * (x-y))

def myHLL(params, seq, sign=1.0, verbose=False):
    """ (Negative) Log-likelihood """
    mu, alpha, beta = params[0], params[1], params[2]
    last = seq[-1]
    constant_survival = last * mu
    updating_survival, occurrence_likelihood, A = 0.0, 0.0, 0.0
    epsilon = 1e-50
    prev = -np.inf
    for curr in seq:
        A = kernel(curr, prev, beta) * A
        updating_survival += (kernel(last, curr, beta) - 1)
        occurrence_likelihood += np.log(mu + alpha * A + epsilon)
        A += 1
        prev = curr
    ll = sign * (-constant_survival + (alpha/beta) * updating_survival + occurrence_likelihood)
    return ll

def myunivariate(seq, verbose=False):
    params = [0.01, 0.03, 0.04]     #[0.01, 0.03, 0.04]
    bounds = [(0, None), (0, None), (0, None)]
    res = minimize(myHLL, params, args=(seq, -1.0, verbose),
                   method="L-BFGS-B",
                   bounds=bounds,
                   options={
                       "ftol": 1e-9, "gtol": 1e-5, "maxls": 50, "maxcor":50,
                       "maxiter": 10000, "maxfun": 10000, "eps": 1e-6, #eps: 1e-6
                       "iprint": -1
                   })
    print(res)
    return res.x[0], res.x[1], res.x[2]
coeffs = myunivariate(time)
HLL = myHLL(coeffs, time)

#intensity
inten = np.zeros(int(np.ceil(time[-1])))
A, prev = 0, 0
for t in range(1, int(np.ceil(time[-1]))):
    prev = time[np.where(time < t)] #time[-1] is not here
    for T_i in prev:
        A += kernel(t, T_i, coeffs[2])
    inten[t] = coeffs[0] + coeffs[1] * A
    A = 0
fig = plt.figure()
plt.ylabel('Intensity')
plt.xlabel('Seasonally adjusted time')
plt.plot(inten)
del A, t, prev

#compensator *dont do*
compen = np.zeros(int(np.ceil(time[-1])))
B, prev = 0, 0
for t in range(1, int(np.ceil(time[-1]))):
    prev = time[np.where(time < t)] #time[-1] is not here
    for T_i in prev:
        B += kernel(t, T_i, coeffs[2]) - 1
    compen[t] = t*coeffs[0] - (coeffs[1]/coeffs[2])*B
    B = 0
fig = plt.figure()
plt.ylabel('Compensator')
plt.xlabel('Time in minutes')
plt.plot(compen)
df = pd.DataFrame(compen)
df.to_csv(path_or_buf='C:\\Users\\user\\Documents\\tweets\\adjusted_compensator.csv', index=False)

#short intensity *dont do*
inten = np.zeros_like(time)
A, prev, t = 0, 0, 0
for curr in time:
    A = kernel(curr, prev, coeffs[2]) * A
    inten[t] = coeffs[0] + coeffs[1] * A
    prev = curr
    A += 1
    t += 1
plt.ylabel('Intensity')
plt.xlabel('Tweets')
plt.plot(inten)
#short compensator
compen = np.zeros_like(time)
B, prev, t = 0, 0, 0
for curr in time:
    prev = time[time < curr]
    for T_i in prev:
        B += kernel(curr, T_i, coeffs[2]) - 1
    compen[t] = curr*coeffs[0] - (coeffs[1]/coeffs[2])*B
    t += 1
    B = 0
plt.plot(compen)
df = pd.DataFrame(compen)
df.to_csv(path_or_buf='C:\\Users\\user\\Documents\\adjusted_compensator.csv', index=False)
