#%%
# Import packages

# Standard
import time
import numpy as np
from scipy.io import loadmat,savemat
from scipy.spatial.distance import hamming
from math import isclose

# Catch exits
import atexit

# For parallelization
from multiprocess import Pool, cpu_count
from sys import platform


#%%
# Classical part
def binary_labels(N):
    """ 
    Return list of all binary numbers smaller than 2**N, starting with 0
    """
    # np.binary_repr(n) would also do it
    return np.array([format(i,'0{}b'.format(N)) for i in range(2**N)])

def probabilities_from_dict(dictionary:dict):
    """
    Returns normalized array, containing the probabilities of all possible binary representations.
    First entry corresponds to 00..0
    """
    # Test whether dictionary is empty
    assert(bool(dictionary)), 'dictionary is empty'

    # Extract number of bits used
    n = len(next(iter(dictionary)))

    # Write all values into an array at position corresponding to key
    output_probabilities = np.zeros(2**n)
    for key,value in dictionary.items():
        output_probabilities[int(key,2)] = value
    
    return output_probabilities/np.sum(output_probabilities)

def new_weights(n:int, m:int, output_x=True,distinct_memories=True):
    """
    Initialize m new n-dimensional patterns and create weighting matrix of Hebbian learning from it

    Args:       - n input dimension
                - m number of stored patterns

    Returns:    - nxn weighting matrix W
                - (if output_x =True) mxn array containing the randomly chosen patterns 
    """
    if not isinstance(n, (int,np.integer)): 
        raise TypeError('n must be int')
    if not isinstance(m, (int,np.integer)):
        raise TypeError('m must be int')
    assert(m != 0), 'Store at least one pattern'

    # Initialize random patterns to memorize
    if distinct_memories:
        if n < 31:
            assert(m <= 2**n), 'cannot store {} distinct patterns when only 2**{} = {} exist in total'.format(m,n,2**n)
        x = np.reshape(initialize_distinct_memories(n,m), (m,1,n))
    else:
        x = np.random.rand(m,1,n)>0.5
    x = np.ones((m,1,n),dtype=int) - 2*x

    # Calculate Hebbian learning matrix from memories
    W = np.zeros((n,n),dtype=int)
    for p in range(m):
        W += x[p].T @ x[p]
    W -= m*np.diagflat(np.ones(n,dtype=int))

    if output_x:
        return W, x[:,0,:]
    else:
        return W

def initialize_distinct_memories(n:int, m:int):
    """
    Randomly generate m memories, which are bit strings of length n.
    The memories are unique, so no memory is contained twice.
    """

    memories = np.zeros((m,n),dtype=bool)
    # Already generate first memory
    memories[0,:] = np.random.rand(n) > 0.5

    # iterate over remaining memories
    for i in range(1,m):
        not_new = True
        while not_new:
            # Generate new memory
            new_memory = np.random.rand(n) > 0.5
            # Test whether new_memory is already contained in memories
            # if yes: repeat. if not: store this memory 
            not_new = np.any(np.all(new_memory == memories[:i,:],axis=1))        

        memories[i] = new_memory

    return memories

def weights_from_memories(memories:np.ndarray):
    """
    Returns nxn weighting matrix of Hebbian learning from patterns that should be stored

    Args:       - memories is mxn array with entries +-1
    """
    assert(len(np.shape(memories))==2), 'Wrong input shape'
    n = np.shape(memories)[1]
    m = np.shape(memories)[0]
    assert(m > 0), 'Store at least one pattern'

    memories = np.reshape(memories, (m,1,n))
    W = np.zeros((n,n),dtype=int)
    for p in range(m):
        W += memories[p].T @ memories[p]
    W -= m*np.diagflat(np.ones(n,dtype=int))

    return W

def binary01_to_pm1(x:np.ndarray):
    """
    Returns +1/-1 representation of given 0/1 binary array. (0 <-> 1), (1 <-> -1)
    """
    return np.ones_like(x,dtype=int) - 2*x 

def binarypm1_to_01(x:np.ndarray):
    """
    Returns 0/1 representation of given +1/-1 binary array. (0 <-> 1), (1 <-> -1)
    """
    return (np.ones_like(x,dtype=int) - x) //2

def int_to_binary01_array(N,width):
    return np.array([int(np.binary_repr(N,width=width)[i]) for i in range(width)],dtype=int)

def synchronous_update(W,x, num_updates=1):
    """
    Updates x in synchronous way: Outputs np.sign(Wx) for num_updates=1, else several times
    """
    if num_updates == 0:
        return x
    update = x.copy()
    for _ in range(num_updates):
        np.sign(W @ update,out=update)
    return update

def asynchronous_update(W,x, num_updates=None):
    """
    Updates x in asynchronous way, so applies np.sign(W_ij x_j) componentwise,
    where component is randomly chosen
    """
    if num_updates is None:
        num_updates = len(x)
    if num_updates == 0:
        return x
    update = x.copy()
    indices = np.random.randint(len(x),size=num_updates)
    for i in range(num_updates):
        np.sign(W[indices[i],:] @ update, out=update[indices[i]])
    return update


def test_error_rate(n:int, m:int, runs=1000, num_updates=1, output_std=True, asynchronous=False, distinct_memories=True,parallel=True,processes=None):
    """
    Estimate average error rate, when requiring that all stored patterns must be retrieved perfectly.
    In each iteration, new stored patterns are generated uniformly at random. 
    
    Args:   - n: input dimension
            - m: number of stored patterns
            - runs: number of iterations for average
            - num_updates: number of synchronous updates, so how often sign(W...) is applied
            - synchronous updates if True, else asynchronous

    Returns:  - Average rate of how often at least one stored pattern cannot be retrieved after num_updates updates
            - Standard deviation if output_std = True
    """
    at_least_one_fail = np.zeros(runs,dtype=bool)

    for k in range(runs):
        W, x = new_weights(n,m,distinct_memories=distinct_memories)
        distances = np.zeros(m)
        for p in range(m):
            if asynchronous:
                updated = asynchronous_update(W,x[p],num_updates=num_updates)
            else:
                updated = synchronous_update(W,x[p],num_updates=num_updates)

            distances[p] = hamming(updated,x[p])*n
            if distances[p] > 0:
                at_least_one_fail[k] = True
                break # if a single error occurs, no need to test all remaining patterns (at least in strict formulation of errors)

    if output_std:
        return np.average(at_least_one_fail), np.std(at_least_one_fail)
    else:
        return np.average(at_least_one_fail)

def flip_locations(n,number_flips):
    # For large n and number_flips close to n/2, better choose a different scheme 
    # and just take any generation + post-selection later
    """
    Return bool array of dimension n with number_flips entries as True entries 
    """
    if not isinstance(n, (int,np.integer)): 
        raise TypeError('n must be int')
    if not isinstance(number_flips, (int,np.integer)): 
        raise TypeError('number_flips must be int')
    locations = np.zeros(n,dtype=bool)
    rate = number_flips/n
    while np.sum(locations) != number_flips:
        locations = (np.random.rand(n)-(1-rate))>0
    return locations

def uniform_noise(data,p,binary01=True):
    """
    Apply uniform noise to data, flip bits with probability p.

    Args:   -data is a binary array with entries 0 or 1 if binary01=True and entries +-1 else
            -p is probability to flip (noise rate)

    Returns noisy copy of data
    """
    assert(p>=0 and p<=1), 'p is not a valid probability'
    corrupted_data = data.copy()

    mask_flip = np.random.random_sample(np.shape(data)) < p 
    np.random

    # flip depending on whether in 0/1 or +1/-1 binary representation
    if binary01:
        corrupted_data[mask_flip] = (corrupted_data[mask_flip]  + 1) % 2
    else:
        corrupted_data[mask_flip] *= -1

    return corrupted_data

def estimate_m_max(n, runs, asynchronous, num_updates, distinct_memories,threshold):
    """
    Estimate the maximum m for which the error rate is exactly zero for a given number of runs 
    """
    m = 0
    error_rate = 0
    while error_rate <= threshold:
        m += 1
        error_rate,_ = test_error_rate(n, m, runs=runs, asynchronous=asynchronous, num_updates=num_updates, 
                                        distinct_memories=distinct_memories)

    return  m - 1


def capacity_estimation(n, threshold=0, runs=1000, num_averages=10, num_updates=1, asynchronous=False,distinct_memories=True,parallel=True,processes=None,use_SER=True):
    """
    Estimates the number m_max of storeable patterns for all n given as array, 
    such that all m smaller or equal to m_max have an error rate which is exactly 0.
    Repeats the estimation of m_max with run samples of the error rate and repeats this num_averages times
    Returns mean of m_max and standard deviation as np.array 
    """
    m_max = -1*np.zeros((len(n),num_averages))

    # loop over all n:
    for i in range(len(n)):
        print(n[i])

        # Parallel method
        if parallel:
        
            # Repeat estimation of at_least_one_fail runs times for averaging in a parallelized fashion
            pool = Pool(processes= processes)
            results = [pool.apply_async(estimate_m_max,
                        args=(n[i],runs, asynchronous,num_updates,distinct_memories,threshold)) for _ in range(num_averages)]
            output = [process.get() for process in results]
            m_max[i,:]= np.array(output,dtype=int)
            pool.terminate()
        
        # Standard method
        else:
            for k in range(num_averages):
                m = 0
                error_rate = 0
                while error_rate <= threshold:
                    m += 1
                    if use_SER:
                        error_rate,_ = test_error_rate(n[i], m, runs=runs, asynchronous=asynchronous, num_updates=num_updates, 
                                                distinct_memories=distinct_memories,parallel=parallel, processes=processes)
                    else:
                        error_rate,_,_,_ = test_error_rate_attraction(n, m[i], 0,
                                                N_test_samples_around_memory=1, averages=1,
                                                asynchronous=asynchronous, num_updates=num_updates, distinct_memories=True,
                                                parallel=parallel, processes=processes)
                m_max[i,k] = m - 1

    return np.mean(m_max,axis=1),np.std(m_max,axis=1)

#%%
def estimate_error_rates_in_parallel(n,m,distinct_memories,N_test_samples_around_memory,noise_rate,asynchronous,num_updates):
    """
    Method to estimate MessageErrorRate and BitErrorRate in parallel
    """
    W, x = new_weights(n,m,distinct_memories=distinct_memories)
    distances = np.zeros((m,N_test_samples_around_memory))

    for p in range(m):
        for i_test in range(N_test_samples_around_memory):
            # create a test sample by applying uniform noise to the corresponding memory
            test_string = uniform_noise(x[p], noise_rate,binary01=False)

            # update the test sample
            if asynchronous:
                updated = asynchronous_update(W,test_string,num_updates=num_updates)
            else:
                updated = synchronous_update(W,test_string,num_updates=num_updates)
            
            # Calculate the distance to the corresponding memory
            distances[p,i_test] = hamming(updated,x[p])*n

    # store the bit error rates
    BitErrorRate = np.mean(distances/n)

    # store message error rates: if any bit is wrong, whole message is wrong!
    MessageErrorRate = np.mean(distances>0)

    return MessageErrorRate, BitErrorRate



def test_error_rate_attraction(n:int, m:int, noise_rate:float, N_test_samples_around_memory=10, averages=1000, 
                                num_updates=1, asynchronous=False, distinct_memories=True,
                                parallel=True, processes=None):
    """
    Estimates message and bit error rates for a given number of test samples in the vacinity of all memories.
    The test samples are produced by flipping every bit with probability noise_rate and the output is after a number 
    of updates is compared to the corresponding memory bit string.
    The whole procedure is repeated averages times, where new stored patterns are generated uniformly at random each time. 
    
    Args:   - n: input dimension
            - m: number of stored patterns
            - noise_rate: noise applied to memories to produce test samples
            - N_test_samples_around_memory: number of noisy test samples produced around each memory
            - averages: number of iterations for averaging, so repeating the whole precedure
            - num_updates: number of synchronous updates, so how often sign(W...) is applied
            - synchronous updates if True, else asynchronous

    Returns:  - mean of MessageErrorRate (MER) estimated for each repetition
            - mean of BitErrorRate (BER) estimated for each repetition
            - standard deviation of MER
            - standard deviation of BER
    """
    BitErrorRates = np.zeros(averages)
    MessageErrorRates = np.zeros(averages)

    # Repeat error rate estimation averages times, each time picking new, randomly chosen memories
    # Parallel method
    if parallel:
        
        # Repeat estimation of at_least_one_fail runs times for averaging in a parallelized fashion
        pool = Pool(processes= processes)
        results = [pool.apply_async(estimate_error_rates_in_parallel,
                    args=(n,m,distinct_memories,N_test_samples_around_memory,noise_rate,asynchronous,num_updates)) 
                    for _ in range(averages)]
        outputs = np.array([process.get() for process in results])
        MessageErrorRates = outputs[:,0]
        BitErrorRates = outputs[:,1]
        pool.terminate()

    # standard method     
    else:
        for k in range(averages):
            W, x = new_weights(n,m,distinct_memories=distinct_memories)
            distances = np.zeros((m,N_test_samples_around_memory))

            for p in range(m):
                # print('memory : {}'.format(x[p]))
                for i_test in range(N_test_samples_around_memory):
                    # create a test sample by applying uniform noise to the corresponding memory
                    test_string = uniform_noise(x[p], noise_rate, binary01=False)
                    # update the test sample
                    if asynchronous:
                        updated = asynchronous_update(W,test_string,num_updates=num_updates)
                    else:
                        updated = synchronous_update(W,test_string,num_updates=num_updates)
                    # print('test: {}, upd: {}'.format(test_string,updated))
                    # Calculate the distance to the corresponding memory
                    distances[p,i_test] = hamming(updated,x[p])*n

            # store the bit error rates
            BitErrorRates[k] = np.mean(distances/n)

            # store message error rates: if any bit is wrong, whole message is wrong!
            # Note difference to strict error definition in test_error_rate:
            # Here: For noise_rate=0, all memories are considered. If one out of all memories cannot be recalled exactly,
            #       the error rate of this round is 1/m
            # Strict:   If anywhere an error occures, assign failure to all. If one out of all memories cannot be recalled
            #           exactly, the error rate of this round is 1!
            # Use np.any(distances>0) to recover the strict definition!
            MessageErrorRates[k] = np.mean(distances>0)

    return np.mean(MessageErrorRates), np.mean(BitErrorRates), np.std(MessageErrorRates), np.std(BitErrorRates)

