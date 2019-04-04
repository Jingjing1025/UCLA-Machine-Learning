import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

mean = [0, 0]
cov = [[1, 0], [0, 1]]  # diagonal covariance
x, y = np.random.multivariate_normal(mean, cov, 1000).T
fig = plt.figure()
plt.plot(x, y, 'x')
fig.savefig('10a.png')

mean_b = [1, 1]
x, y = np.random.multivariate_normal(mean_b, cov, 1000).T
fig = plt.figure()
plt.plot(x, y, 'x')
fig.savefig('10b.png')

cov_c = [[2, 0], [0, 2]] 
x, y = np.random.multivariate_normal(mean, cov_c, 1000).T
fig = plt.figure()
plt.plot(x, y, 'x')
fig.savefig('10c.png')

cov_d = [[1, 0.5], [0.5, 1]] 
x, y = np.random.multivariate_normal(mean, cov_d, 1000).T
fig = plt.figure()
plt.plot(x, y, 'x')
fig.savefig('10d.png')

cov_e = [[1, -0.5], [-0.5, 1]] 
x, y = np.random.multivariate_normal(mean, cov_e, 1000).T
fig = plt.figure()
plt.plot(x, y, 'x')
fig.savefig('10e.png')