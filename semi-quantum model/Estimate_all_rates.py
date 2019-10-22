#%%
# Import packages

# Standard
import time
import numpy as np
from scipy.io import savemat

# Import functions for simulation
from function_collection import *


#%%  
# General settings

shots = 0   # if 0: Probabilities extracted from density matrices. 
            # if >0: Probabilities estimated on a number of simulated experiments, using Monte-Carlo simulation. 
n_bits = 4  # input size
runs = 500  # repetitions to get point estimates and standard error for error rates
distinct_memories = True # Whether memory patterns must be distinct or not

stop =  2**n_bits + 1  # n+1
m = np.arange(start=1,stop=stop,dtype=int)


# Simulation-related settings
use_QI_backend = False      # switch between Quantum Inspire and default simulator of ProjectQ as simulator back-ends
gate_fusion = False         # only required for ProjectQ default simulator
parallel = True             # parallel estimation of several repetitions
processes = 6               # make dependend on accessible RAM! 

verbose = True
separate_outputs = True     # if True: circuit for output qubits are simulated separately, requiring only n+1 qubits

# Output files 
save_data_path  = './Results/all_rates/19_09_30_n{}_runs{}_shots{}_distinct.mat'.format(n_bits,runs,shots)
logfile_name    = './Results/all_rates/19_09_30_n{}_runs{}_shots{}_distinct_log.txt'.format(n_bits,runs,shots)
errorlog_name   = './Results/all_rates/19_09_30_n{}_runs{}_shots{}_distinct_errorlog.txt'.format(n_bits,runs,shots)

#%%
# error rate estimation

# Initialize arrays
probabilities_means = []
probabilities_stds = []
SER = np.zeros(stop-1)
SER_std = np.zeros(stop-1)
MER = np.zeros(stop-1)
MER_std = np.zeros(stop-1)
BER = np.zeros(stop-1)
BER_std = np.zeros(stop-1)

# write settings on log-file
with open (logfile_name,'a') as f:
    f.write('Estimate all error rates: SER, MER, BER\n')
    f.write('m evaluated from 1 to {}\n'.format(stop-1))
    f.write('use_QI_backend = {}\n'.format(use_QI_backend))
    f.write('parallel = {}\n'.format(parallel))
    f.write('processes = {}\n'.format(processes))
    f.write('OMP_NUM_THREADS = 1\n')
    f.write('Start: {}\n'.format(time.strftime('%a, %d.%m.%y, %H:%M:%S')))

# calculate error rates
start_time = time.time()
for i in range(0,stop-1):
    with open (logfile_name,'a') as f:
        f.write('{} / {}  {}\n'.format(i+1,stop-1,time.strftime('%d.%m.%y, %H:%M:%S')))

    SER[i],SER_std[i],MER[i],MER_std[i],BER[i],BER_std[i],probabilities = test_error_rate_quantum_parallel_attraction(n_bits,
                                                            m[i],runs=runs,use_QI_backend=use_QI_backend,distinct_memories=distinct_memories,
                                                            separate_outputs=separate_outputs,gate_fusion=gate_fusion,
                                                            output_probabilities=True,output_std=True, verbose=verbose,
                                                            logfile_name=logfile_name,errorlog_name=errorlog_name,processes=processes,
                                                            shots=shots)


    probabilities_means.append(np.average(probabilities))
    probabilities_stds.append(np.std(probabilities))

    # Save data in each iteration
    savemat(save_data_path, 
        {'m': m,
        'SER': SER,
        'SER_std': SER_std,
        'MER': MER,
        'MER_std': MER_std,
        'BER': BER,
        'BER_std': BER_std,
        'probabilities_means': probabilities_means,
        'probabilities_stds': probabilities_stds})

end_time = time.time()

duration = (end_time-start_time)/60     # [min]
with open (logfile_name,'a') as f:
    f.write('Total duration: {} h {:.2f} min\n'.format(duration//60,duration%60))
