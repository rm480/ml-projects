#hawkes process



from tick.hawkes import HawkesExpKern, HawkesEM
from tick.plot import plot_hawkes_kernels
import pandas as pd
time = pd.read_csv('timestamp.csv').to_numpy().transpose().flatten().astype(float)
temp = list()
temp.append(time)
time = temp

beta=0.5
hawkes_learner = HawkesExpKern(beta, gofit='least-squares', C=1000, tol=1e-10, max_iter=10000)
hawkes_learner.fit(time)
hawkes_learner.plot_estimated_intensity(time)

hawkes_learnerEM = HawkesEM(3000, kernel_size=3000, tol=1e-10, max_iter=10000, n_threads=4)
hawkes_learnerEM.fit(time)
print(hawkes_learnerEM.baseline)
plot_hawkes_kernels(hawkes_learnerEM)
