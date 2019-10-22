#%%
# import packages
import numpy as np 
from scipy.io import savemat,loadmat
from scipy.special import binom
from itertools import combinations

from functions_for_creation import *


#%%
# Setup and create datasets, only training set and memories

n = 5
m_stop = 7

distance = 1
fraction = 1.
rounds = 50
mix_data = False
distinct_memories = True

#%%
# generate training (and test) sets

# store_training_data(n, rounds, m_stop=m_stop, distance=distance, fraction=fraction, 
#                         mix_data=mix_data, distinct_memories=distinct_memories)

N_vicinity = 100
store_training_and_test_data(n, rounds, N_vicinity, m_stop=m_stop, distance=distance, fraction=fraction, 
                        mix_data=mix_data, distinct_memories=distinct_memories)




#%%
n = 4
rounds = 100
fraction = 1
distance = 1
distinct_memories = True
m_stop = 2**n

save_file_name = './results/combined_rates/contradiction_rates_n{}_r{}_d{}.mat'. format(n,rounds,distance)


# Initialize contradiction rates
contradiction_rates = np.zeros(m_stop)
contradiction_rates_std = np.zeros(m_stop)

for m in range(1,m_stop+1):
    # generate training sets
    print('m={}'.format(m))
    training_data, memories = create_trainingsets(n, m, rounds, distance=distance, fraction=fraction, distinct_memories=distinct_memories)
    N_training_samples = np.shape(training_data)[1]
    print('trainingset generated')

    # create labels for each input training sample = corresponding memory (not needed actually)
    labels =np.zeros((rounds,N_training_samples),dtype=int)
    for r in range(rounds):
        for i in range(m):
            for j in range(N_training_samples//m):
                labels[r,j+i*N_training_samples//m] = binary01_array_to_int(memories[r,i])
    print('labels created')

    # count the contradiction errors
    training_data_int = translate_to_comp_basis_state(training_data)[0]
    working_equations = -np.ones((rounds, 2**n))
    contradiction_errors = np.zeros(rounds)

    for r in range(rounds):
        for i in range(N_training_samples):
            if working_equations[r,training_data_int[r,i]] < 0:
                working_equations[r,training_data_int[r,i]] = labels[r,i]
            elif working_equations[r,training_data_int[r,i]] != labels[r,i]:
                contradiction_errors[r] +=1

    contradiction_rates[m-1] = np.mean(contradiction_errors)/N_training_samples
    contradiction_rates_std[m-1] = np.std(contradiction_errors)/N_training_samples


print(contradiction_rates)
print(contradiction_rates_std)


# savemat(save_file_name, {'contradiction_rates': contradiction_rates,
#                         'contradiction_rates_std': contradiction_rates_std})



# ---------------------------------------------------------------------------
# Testing and checks 
#%%
# Check some things 
# data_dict = loadmat('./data/trainingset_n3_rounds20_m3_d1.mat')
data_dict = loadmat('./data_vicinity/trainingset_n3_rounds100_Nvic100_m3_d1_distinct.mat')
phi_out = np.squeeze(data_dict['phi_out'])
phi_in = np.squeeze(data_dict['phi_in'])
memory_states = data_dict['memory_states']
memory_int_representation = data_dict['memory_int_representation']
test_states_noise10 = data_dict['test_states_noise10']
test_states_noise20 = data_dict['test_states_noise20']
test_states_noise30 = data_dict['test_states_noise30']

#%%
for p in range(3):
    print('memory = {}'.format(memory_states[0,:,p]))
    for j in range(5):
        print(test_states_noise10[0,:,p*N_vicinity+j])

#%%
# estimate fraction beta, such that training set remains the same
f = lambda n: n*(n+1)**2 /2
g = lambda n: n**2*(n**2-1)/4 + n*(n+1)**2/2

n = 4
print('beta_{} = {}'.format(n,f(n)/g(n)))
print('dist 1: {} samples'.format(f(n)))
print('dist 2: {} samples'.format(g(n)))
#%%
print(np.shape(phi_out))
print(np.shape(memory_states))
print(np.shape(memory_int_representation))

#%%
print('in')
print(phi_in[0,:,10])
print('out')
print(phi_out[0,:,10])


# Test Cases ----------------------------------------------------------------------
#%%
m = 3
# Create dataset only for one round and n,m fix
training_data, memories = create_single_trainingset(n, m, distance=distance, fraction=fraction)
print(memories.astype(int))
print(training_data.astype(int))

#%%
# Create dataset for several rounds and n,m fix
training_data, memories = create_trainingsets(n, m, rounds, distance=distance, fraction=fraction)
print(memories.astype(int))
# print(training_data.astype(int))

#%%
# Test translation of binary arrays to computational basis states
#  
# print(memories.astype(int))
int_representations, state_vectors = translate_to_comp_basis_state(memories)

# print(int_representations.astype(int))
print(state_vectors.astype(int))


#%%
a = np.zeros((3,5))
for i in range(5):
    a[:,i] = i* np.ones(3)

print(a)

permutation = np.random.permutation(np.arange(5))

print(a[:,permutation])