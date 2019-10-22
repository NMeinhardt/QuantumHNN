#%%
import numpy as np 
from scipy.io import savemat
import time

from classical_function_collection import *


#%%
# Settings
n = 5
num_updates = 1
runs = 10000
asynchronous = False
distinct_memories = False

stop =  2**n  # 2**n
m = np.arange(start=1,stop=stop+1,dtype=int)

#%%
# Calculate error rate
start_time = time.time()

# Initialize arrays
SER = np.zeros(stop)
SER_std = np.zeros(stop)

# estimate SER for m = 1,...,stop
for i in range(stop):
    print(m[i])

    SER[i],SER_std[i] = test_error_rate(n,m[i],runs=runs ,asynchronous=asynchronous, 
                                    num_updates=num_updates, distinct_memories=distinct_memories)
end_time = time.time()

print('Duration: {} s '.format(end_time-start_time))


#%%
# Save data
savemat('./direct_retrieval/19_09_26_n{}_runs{}_{}upds_not_distinct.mat'.format(n,runs,num_updates), 
    {'m': m,
    'SER': SER,
    'SER_std': SER_std})



