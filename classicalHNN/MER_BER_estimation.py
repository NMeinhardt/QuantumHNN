#%%
import numpy as np 
from scipy.io import savemat
import time

from classical_function_collection import *


#%%
# Settings
n = 20
num_updates = 1
averages = 10000
asynchronous = False
distinct_memories = True

m_min =  1
m_max = 20  # 2**n 
noise_rate = 0.1
N_test_samples_around_memory = 1

parallel = False
processes = 4

m = np.arange(start=m_min,stop=m_max+1,dtype=int)


#%%
# Estimate error rates (MER and BER)
start_time = time.time()
MER_rates = np.zeros(m_max)
MER_rates_std = np.zeros(m_max)
BER_rates = np.zeros(m_max)
BER_rates_std = np.zeros(m_max)

for i in range(len(m)):
    print(m[i])
    MER_rates[i], BER_rates[i], MER_rates_std[i], BER_rates_std[i] = test_error_rate_attraction(n, m[i], noise_rate,
                                                N_test_samples_around_memory=N_test_samples_around_memory, averages=averages,
                                                asynchronous=asynchronous, num_updates=num_updates, distinct_memories=distinct_memories,
                                                parallel=parallel, processes=processes)
end_time = time.time()

print('Duration: {:.2f} s '.format(end_time-start_time))


#%%
# Save data
savemat('./vicinity/19_09_09_n{}_averages{}_noise{}_Nvic{}_upds{}_distinct.mat'.format(n, averages, noise_rate,
                                                                                    N_test_samples_around_memory,num_updates), 
    {'m': m,
    'MER_rates': MER_rates,
    'MER_rates_std': MER_rates_std,
    'BER_rates': BER_rates,
    'BER_rates_std': BER_rates_std})



