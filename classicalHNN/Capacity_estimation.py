#%%
# Import 
import numpy as np 
import matplotlib.pyplot as plt
from scipy.spatial.distance import hamming
from scipy.io import loadmat,savemat
import time

from classical_function_collection import capacity_estimation

#%%
# Settings
n = np.arange(2,100)

num_updates = 1
runs = 10000
num_averages = 100
asynchronous = False 
distinct_memories = True
parallel = True
use_SER = False
processes = 4

threshold = 0

#%%
# Estimate capacities
start_time = time.time()
m_max_mean,m_max_std = capacity_estimation(n,threshold=threshold,runs=runs,num_averages=num_averages,num_updates=num_updates, asynchronous=asynchronous,
                                            distinct_memories=distinct_memories, parallel=parallel, processes=processes, use_SER=use_SER)
end_time = time.time()
duration = (end_time-start_time) / 60
print('in s: {:.1f}'.format(duration*60))
print('Duration: {} h {} min '.format(int(duration//60),int(duration%60)))

#%%
# Save data
savemat('./capacity/19_09_30_capacity_runs{}_avgs{}_upds{}_threshold{}_distinct_MER.mat'.format(runs,num_averages,num_updates,threshold), 
    {'n': n,
    'm_max_mean': m_max_mean,
    'm_max_std': m_max_std})





