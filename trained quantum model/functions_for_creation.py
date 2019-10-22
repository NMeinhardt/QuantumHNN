#%%
# import packages
import numpy as np 
from scipy.io import savemat,loadmat
from scipy.special import binom

from itertools import combinations

#%%
# Functions 

def binary01_array_to_int(binary_array):
    """
    Takes a binary array and returns accoring integer.  
    Array must be bool or int1

    Note: First entry is most significant digit.
    """
    out = 0
    for bit in binary_array.astype(bool):
        out = (out << 1) | bit
    return out

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


def create_single_trainingset(n:int, m:int, distance:int=1, fraction=1.,distinct_memories=True):
    """
    Creates a dataset of m n-dim memories, including their neighborhoods within given distance.
    Of this distance, a given fraction of samples is considered. 
    If fraction=1(default), the whole neighborhood is taken into account.

    Returns:    - training data, including memories and samples from neighborhood 
                    of shape (m*(N_samples_around_memories+1), n), where N_samples_around_memories depends on distance and fraction
                - memories of shape (m,n)


    Note: number of samples per memory is fraction*(N_neighbors within given distance),
    rounded to next lower integer. So fraction should be seen as maximum fraction
    """
    assert(distance <= n), 'Distance must not exceed n'

    # Initialize random patterns to memorize
    if distinct_memories:
        assert(m <= 2**n), 'there are not more than 2**n distinct patterns'
        memories = initialize_distinct_memories(n,m)
    else:
        memories = np.random.rand(m,n)> 0.5

    # Get number of samples in neighborhood, depending on total number of neighbors within given distance
    # and the chosen fraction that should be taken into account. 
    N_neighbors = 0
    for d in range(1,distance+1):
        N_neighbors += binom(n,d)
    N_neighbors = int(N_neighbors)
    N_samples_around_memories = int(N_neighbors * fraction)

    # initialize output array
    N_data = m* (N_samples_around_memories+1)
    training_data = np.empty((N_data,n),dtype=bool)

    # produce neighborhoods 
    for l in range(m):
        neighbors = np.ones((N_neighbors,n),dtype=bool)* memories[l]
        i = 0
        for d in range(1,distance+1):
            iterator = combinations(np.arange(n),d)
            swap_positions = np.array(list(iterator))
            for swap in swap_positions:
                neighbors[i][swap] = ~neighbors[i][swap]
                i += 1

        # store memory 
        training_data[l*(N_samples_around_memories+1)] = memories[l]
        
        # pick N_samples_around_memories samples of neighborhood at random and store
        pick_samples = np.random.choice(np.arange(N_neighbors), size=N_samples_around_memories, replace=False)
        for i in range(N_samples_around_memories):
            training_data[l*(N_samples_around_memories+1)+1+i] = neighbors[pick_samples[i]]
    
    return training_data, memories

def create_trainingsets(n:int, m:int, rounds:int, distance:int=1, fraction=1.,distinct_memories=True):
    """
    Creates dataset for a given number of rounds of training. 
    Returns:
        - Trainingsets of shape (rounds, m*(N_samples_around_memories+1), n).
        - memories of shape (rounds, m, n)
    """
    training_data_all = []
    memories_all = []
    for _ in range(rounds):
        data, memories = create_single_trainingset(n, m, distance=distance, fraction=fraction, distinct_memories=distinct_memories)
        training_data_all.append(data),
        memories_all.append(memories)
    
    return np.array(training_data_all,dtype=bool), np.array(memories_all,dtype=bool)


def translate_to_comp_basis_state(binary_array:np.ndarray):
    """
    Returns:
        - int representation of last dimension of binary_array, all other dimensions remain unaffected
        - corresponding computational basis states as state vector of dimension 2**(last dimension of binary array)
    
    Note: first element of state vectors correspond to |0>, so are the least significant
    """
    # Store shape of input
    shape = np.shape(binary_array)
    if len(shape) == 1:
        return binary01_array_to_int(binary_array)

    # reshape inputs, such that last dimension can be translated to an integer
    n_previous_dim = np.prod(shape[:-1])
    reshaped_array = np.reshape(binary_array, (n_previous_dim,-1))

    # initialize integer representations of bit strings in last dimension of binary_array
    output_int = np.empty(n_previous_dim,dtype=int)
    # initialize state representations of bit strings in last dimension of binary_array, states are of dim 2**n
    state_vector_representations = np.zeros((n_previous_dim, 2**shape[-1]),dtype=bool)
    # calculate integers from bit strings and afterwards the corresponding amplitudes
    for i in range(n_previous_dim):
        output_int[i] = binary01_array_to_int(reshaped_array[i])
        state_vector_representations[i,output_int[i]] = True

    # reshape outputs
    output_int = np.reshape(output_int, shape[:-1])
    state_vector_representations = np.reshape(state_vector_representations, shape[:-1]+(2**shape[-1],))
    # return both integer representations and the corresponding computational basis states explicitly
    
    return output_int.astype(int), state_vector_representations


def store_training_data(n:int, rounds:int, m_stop:int=None, distance:int=1, fraction=1., mix_data=False, distinct_memories=True):
    """
    Create and store training data for all 1 <= m <= n and store them in separate .mat files.
    The training data are directly converted to the right shape to use the matlab code
    """
    if not isinstance(m_stop, (int,np.integer)):
        m_stop = n

    for m in range(1,m_stop+1):
        print('m = {}'.format(m))
        # create memories uniformly at random and neighboring states depending on distance and fraction
        # repeat rounds-times -> memories are of shape (rounds, m, n)
        training_data, memories = create_trainingsets(n, m, rounds, distance=distance, fraction=fraction, distinct_memories=distinct_memories)

        # convert bit strings to corresponding integers or state vectors
        memory_int_representation, state_vectors_memories = translate_to_comp_basis_state(memories)
        int_representations_all_data, state_vectors_all_data = translate_to_comp_basis_state(training_data)

        # prepare phi_out of the same shape as state_vectors_all_data, which contain only copies of the m memories
        N_samples = np.shape(int_representations_all_data)[-1]
        phi_out = np.zeros_like(state_vectors_all_data)
        for i in range(m):
            for j in range(N_samples//m):
                phi_out[...,j+i*N_samples//m,:] = state_vectors_memories[...,i,:]


        # For Matlab code shape should be (rounds, 1..2**n entries, samples), but currently it is (rounds, samples, 1..2**n entries)
        # Thus swap axes 1 and 2
        state_vectors_all_data = np.swapaxes(state_vectors_all_data,1,2)
        phi_out = np.swapaxes(phi_out,1,2)
        state_vectors_memories = np.swapaxes(state_vectors_memories,1,2)

        # print('m={}'.format(m))
        # print('inputs')
        # print(state_vectors_all_data.astype(int))
        # print('labels')
        # print(phi_out.astype(int))
        
        # Mix the training sets if mixed=True
        if mix_data:
            for r in range(rounds):
                # create a new permutation in each round 
                permutation = np.random.permutation(np.arange(N_samples))
                state_vectors_all_data = state_vectors_all_data[...,:,permutation]
                phi_out = phi_out[...,:,permutation]


        # store data 
        if mix_data:
            file_name = './data/trainingset_n{}_rounds{}_m{}_d{}_mixed.mat'.format(n,rounds,m,distance)
        elif distinct_memories:
            file_name = './data/trainingset_n{}_rounds{}_m{}_d{}_distinct.mat'.format(n,rounds,m,distance)
        else:
            file_name = './data/trainingset_n{}_rounds{}_m{}_d{}.mat'.format(n,rounds,m,distance)

        savemat(file_name, {'phi_in': state_vectors_all_data.astype(float),
                            'phi_out': phi_out.astype(float), 
                            'N_samples': N_samples,
                            'memory_states': state_vectors_memories.astype(float),
                            'memory_int_representation': memory_int_representation.astype(float)})


def generate_test_set(memories, N_vicinity:int, noise_rate:float):
    """
    Returns a test set test_set of shape (rounds, 2**n, m*N_vicinity) for the input memories 
    of shape (rounds, m, n).  Note that the latter are the actual bit strings!
    """
    rounds,m,n = np.shape(memories)

    # initialize test set
    test_set = np.zeros((rounds, 2**n, m*N_vicinity))

    # repeat for every round
    for r in range(rounds):
        # create a vicinity of N_vicinity noisy patterns around each memory
        for p in range(m):
            for j in range(N_vicinity):
                # Apply noise to bit string
                noisy_pattern = np.reshape(uniform_noise(memories[r,p,:],noise_rate),(1,-1))
                # Translate to computational basis
                _, noisy_pattern_comp_basis = translate_to_comp_basis_state(noisy_pattern)
                # Store all in one array 
                
                test_set[r,:,p*N_vicinity+j] = noisy_pattern_comp_basis[0]
        #         if j==0:
        #             break
        #     if p==0:
        #         break
        # if r==0:
        #     break   

    # noisy_pattern_comp_basis = np.swapaxes(noisy_pattern_comp_basis,1,2)
    return test_set


def store_training_and_test_data(n:int, rounds:int, N_vicinity:int, noise_rates=[0.1,0.2,0.3], m_stop:int=None,
                                distance:int=1, fraction=1., mix_data=False, distinct_memories=True):
    """
    Create and store training data for all 1 <= m <= n and store them in separate .mat files.
    The training data are directly converted to the right shape to use the matlab code
    """
    if not isinstance(m_stop, (int,np.integer)):
        m_stop = n

    for m in range(1,m_stop+1):
        print('m = {}'.format(m))
        # create memories uniformly at random and neighboring states depending on distance and fraction
        # repeat rounds-times -> memories are of shape (rounds, m, n)
        training_data, memories = create_trainingsets(n, m, rounds, distance=distance, fraction=fraction, distinct_memories=distinct_memories)

        # convert bit strings to corresponding integers or state vectors
        memory_int_representation, state_vectors_memories = translate_to_comp_basis_state(memories)
        int_representations_all_data, state_vectors_all_data = translate_to_comp_basis_state(training_data)

        # prepare phi_out of the same shape as state_vectors_all_data, which contain only copies of the m memories
        N_samples = np.shape(int_representations_all_data)[-1]
        phi_out = np.zeros_like(state_vectors_all_data)
        for i in range(m):
            for j in range(N_samples//m):
                phi_out[...,j+i*N_samples//m,:] = state_vectors_memories[...,i,:]


        # For Matlab code shape should be (rounds, 1..2**n entries, samples), but currently it is (rounds, samples, 1..2**n entries)
        # Thus swap axes 1 and 2
        state_vectors_all_data = np.swapaxes(state_vectors_all_data,1,2)
        phi_out = np.swapaxes(phi_out,1,2)
        state_vectors_memories = np.swapaxes(state_vectors_memories,1,2)

        # print('m={}'.format(m))
        # print('inputs')
        # print(state_vectors_all_data.astype(int))
        # print('labels')
        # print(phi_out.astype(int))
        
        # Mix the training sets if mixed=True
        if mix_data:
            for r in range(rounds):
                # create a new permutation in each round 
                permutation = np.random.permutation(np.arange(N_samples))
                state_vectors_all_data = state_vectors_all_data[...,:,permutation]
                phi_out = phi_out[...,:,permutation]

        # store training data 
        if mix_data:
            file_name = './data_vicinity/trainingset_n{}_rounds{}_Nvic{}_m{}_d{}_mixed.mat'.format(n,rounds,N_vicinity,m,distance)
        elif distinct_memories:
            file_name = './data_vicinity/trainingset_n{}_rounds{}_Nvic{}_m{}_d{}_distinct.mat'.format(n,rounds,N_vicinity,m,distance)
        else:
            file_name = './data_vicinity/trainingset_n{}_rounds{}_Nvic{}_m{}_d{}.mat'.format(n,rounds,N_vicinity,m,distance)

        variable_dict = {'phi_in': state_vectors_all_data.astype(float),
                        'phi_out': phi_out.astype(float), 
                        'N_samples': N_samples,
                        'memory_states': state_vectors_memories.astype(float),
                        'memory_int_representation': memory_int_representation.astype(float)}
        

        # generate test data and add to dictionary 
        for gamma in noise_rates:
            variable_dict['test_states_noise{}'.format(int(100*gamma))] = generate_test_set(memories,
                                                                                        N_vicinity,gamma)

        savemat(file_name, variable_dict)
