#%%
# Import packages

# Standard
import time
import numpy as np
from scipy.io import loadmat,savemat
from scipy.spatial.distance import hamming
from math import isclose

# Login required by Quantum Inspire
from coreapi.exceptions import ErrorMessage

# Quantum Simulation, Quantum Inspire has backends for ProjectQ
from projectq import MainEngine
from projectq.backends import ResourceCounter, CircuitDrawer, Simulator
from projectq.cengines import ManualMapper
from projectq.meta import Control, Loop
from projectq.ops import CNOT, All, H, Measure, X,Y,Z, Rx, Rz, Ry
from projectq.setups import restrictedgateset, default

from quantuminspire.api import QuantumInspireAPI
from quantuminspire.projectq.backend_qx import QIBackend

# Catch exits
import atexit

# For parallelization
from multiprocess import Pool
from sys import platform

#%%
# functions for classical part
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


#%%
# Quantum part
def create_engine(use_QI_backend=False, api=None, num_runs=1, verbose=False, gate_fusion=False):
    """
    Creates a new MainEngine. 
    If use_QI_backend= True, Quantum Inspire is used as backend,
    else the ProjectQ default Simulator

    Returns engine for simulation
    Note: backend can be accessed via engine.backend both for QuantumInspire and ProjectQ simulators
    """
    if use_QI_backend:
        assert(api is not None), 'api must be defined if QI-backend should be used'

    # Set up compiler engines
    compiler_engines = default.get_engine_list()
    compiler_engines.extend([ManualMapper(lambda x: x)])

    if use_QI_backend:
        qi_backend = QIBackend(quantum_inspire_api=api,num_runs=num_runs)
        engine = MainEngine(backend=qi_backend, engine_list=compiler_engines,verbose=verbose)
    else: 
        sim = Simulator(gate_fusion=gate_fusion)
        engine = MainEngine(backend=sim,engine_list=compiler_engines,verbose=verbose) 

    return engine


def dict_2keys_to_array(dictionary):
    """ 
    Returns array of length 2 with probability of state |0> as first entry.

    Arg: Dictionary contains only 2 entries and one has key '00..0' or only one entry. 

    If get_probabilities in QIBackend for ProjectQ is changed, adapt this function
    """ 
    assert(len(dictionary)==2 or len(dictionary)==1), 'Wrong size of dictionary'
    prob_as_array = np.zeros(2)
    for x,y in dictionary.items():
        if int(x)==False:
            prob_as_array[0] = y
        else:
            prob_as_array[1] = y

    # Rescale to ensure probability conservation
    return prob_as_array/sum(prob_as_array)


def single_outcomes_to_register_outcome(single_probabilities):
    """
    Returns array with probabilities of all possible string combinations based on the probabilities of each single output qubit.
    The first entry corresponds to |0>, the last to |n-1>

    Args: single_probabilities is an array of shape (n,2)
    """
    try:
        n = len(single_probabilities)
    except:
        raise TypeError('input array must not be empty')
    assert(np.shape(single_probabilities)==(n,2)),'wrong shape of array'

    register_probabilities = np.zeros(2**n)
    for i in range(2**n):
        indices = int_to_binary01_array(i,n)
        
        # probability of a bit string is product of probablities of 0 and 1 for each separate bit
        prob = 1
        for k in range(n):
            prob *= single_probabilities[k,indices[k]]
        register_probabilities[i] = prob
    
    # Make sure that a probability distribution is returned
    return register_probabilities/np.sum(register_probabilities)


# Circuit implementations
def state_preparation(quantum_register, bit_string,verbose=False):
    """
    Initializes quantum state according to classical input.

    Args:   
        - Quantum register of ProjectQ
        - bit_string is a binary array with entries +-1
    
    """
    assert(len(quantum_register) == len(bit_string)), 'Must have the same length!'
    assert(np.all(bit_string**2)), 'Bit string contains entries other than +-1'

    # Transform entries (+1 -> 0), (-1 -> 1)
    bit_string = (0.5*(1-bit_string)).astype(int)
    
    if verbose:
        print('Classical input vector is {}'.format(bit_string))
    for i_bit in range(len(bit_string)):
        if bit_string[i_bit]:
            X | quantum_register[i_bit]

def U_ZX_circuit(engine,quantum_register,weighting_matrix):
    """ 
    Applies whole circuit to quantum register, consisting of all unitaries e^(i theta Z_jxX_i) for all i,j,
    excluding self-connections, so i!=j.

    Args:
        - engine is a valid engine of ProjectQ
        - quantum_register is a non-empty quantum register with an even number of qubits >=4
          The first half contains the input-qubits, the second half the output-qubits
        - weighting matrix is the one from Hebbian learning, controlling the gate parameters
        
    """
    n_qubits = len(quantum_register)
    assert(n_qubits % 2 == 0), 'Number of qubits must be even'
    n_bits = n_qubits//2
    assert(n_bits>1), 'n_bits is too small, dividing by (n-1) causes troubles'
    
    # Apply H to all output-qubits before and after CNOTs/rotations, the H in the middle of circuit cancle each other out
    All(H) | quantum_register[n_bits:]

    # Factor for all angles, multiply with weighting_matrix[i,j] 
    a = np.pi/4 /(n_bits-1)

    # Loop over all output qubits
    for i in range(n_bits):
        # Apply all gates and discard self-connection 
        # From 0 until i-1
        for j in range(i):
            theta = a * weighting_matrix[i,j]
            CNOT | (quantum_register[j],quantum_register[i+n_bits])
            Rz(2*theta) | quantum_register[i+n_bits]
            CNOT | (quantum_register[j],quantum_register[i+n_bits])
        # From i+1 onwards 
        for j in np.arange(i+1,n_bits):
            theta = a * weighting_matrix[i,j]
            CNOT | (quantum_register[j],quantum_register[i+n_bits])
            Rz(2*theta) | quantum_register[i+n_bits]
            CNOT | (quantum_register[j],quantum_register[i+n_bits])
    
    # Apply H to all output-qubits before and after CNOTs/rotations, the H in the middle of circuit cancle each other out
    All(H) | quantum_register[n_bits:]


def most_likely_outcome(probabilities:np.ndarray, output_index=True):
    """
    Returns the bit string corresponding to the most likely outcome, given an array of probabilities.

    Args:       - probabilities is a 1d-array of length 2**n for some n
                - If output_index=True, the index of probabilities corresponding to the highest probability is returned

    Returns:    - int 1d-array containing entries +-1, which corresponds to most likely pattern. 
                If more than one pattern share the highest probability, all these k patterns are returned as (k,2**n)-array.
                - 1d-array containing index of probabilities array corresponding to this probability.
                If more than one pattern share the highest probability, all corresponding indices are returned.
    """
    n = np.log2(len(probabilities))
    assert(n%1 == 0), 'Array is not of length 2**n'
    n = int(n)
    assert(len(probabilities) == np.shape(probabilities)[0]), 'Array must be 1d'

    # Find argmax of probabilities, if maximum probability occures several times the first occurence is stored
    indices = []
    indices.append(np.argmax(probabilities))
    
    # Seach for other occurences of the same probability (can only be at a larger index)
    # It may happen that probabilities are extremely close to each other 
    # -> Thus need to seach again through all indices and compare with relative tolerance
    for i in range(indices[0]):
        if isclose(probabilities[i],probabilities[indices[0]],rel_tol=1e-5):
            indices.append(i)
    for i in np.arange(indices[0]+1,2**n):
        if isclose(probabilities[i],probabilities[indices[0]],rel_tol=1e-5):
            indices.append(i)

    outcomes = np.array([binary01_to_pm1(int_to_binary01_array(i,n)) for i in indices])

    return np.squeeze(outcomes), np.array(indices,dtype=int)


def quantumHNN_update(W:np.ndarray, query:np.ndarray, use_QI_backend=False, api= None, gate_fusion=False, verbose=False, 
                    num_runs=1, return_all_outcomes=False, return_probabilities=False,shots=0):
    """
    Sets up and evaluates quantum circuit of feed-forward like implementation of HNN 
    for a given weighting matrix and query. 
    The simulation is performed either on QuantumInspire (externally) or ProjectQ (locally) simulators

    Args:
        - W is a 2d-array as weighting matrix
        - query is a 1d-array with entries +-1
        - if use_QI_backend = True, QuantumInspire is used, default is the ProjectQ Simulator
        - api to connect to QuantumInspire
        - num_runs is the parameter of QuantumInspire
        - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.
    
    Returns:
        - most likely outcomes (If return_all_outcomes=True, all outcomes with maximum probability are returned.
        Else, only one outcome is returned. In the latter case, a zero-array is returned if multiple outcomes have same probability.)
        - If return_probabilities=True, also the according probabilities are returned
    """
    # Check inputs
    assert(query.ndim == 1), 'Query array is of wrong shape'
    assert(np.all(query**2)), 'Query contains entries other than +-1'
    assert(W.ndim == 2), 'W is of wrong shape'
    n_bits = len(query)
    assert(np.shape(W)[0] == n_bits), 'Sizes of W and query do not match'

    # Allocate quantum register
    eng = create_engine(use_QI_backend=use_QI_backend, api=api, gate_fusion= gate_fusion, num_runs=num_runs,verbose=verbose)
    qureg = eng.allocate_qureg(2*n_bits)

    # Initialize input register according query sting
    state_preparation(qureg[:n_bits],query)

    # Initialize all output qubits as |+i>
    All(Rx(-np.pi/2)) | qureg[n_bits:]

    # Run actual ciruit
    U_ZX_circuit(eng,qureg,W)

    # Flush whole circuit down the pipeline     
    eng.flush()

    # Extract outcome probabilities of Y-measurement of circuit
    if use_QI_backend:
        output_dict = eng.backend.get_probabilities(qureg[n_bits:])
        probabilities = probabilities_from_dict(output_dict)
    else: 
        probabilities = get_probabilities_from_simulator(eng,qureg[n_bits:],shots=shots)

        # Make sure to be in computational basis state before deallocating
        All(Measure) |qureg 

    # Get outcomes with highest probability
    outcomes,indices = most_likely_outcome(probabilities, output_index=True)


    # If return_all_outcomes=True: Return all outcomes, no matter whether a single or multiple outcomes possible
    # If return_probabilities=True, also the according probabilities are returned
    if return_all_outcomes:
        if return_probabilities:
            return outcomes, probabilities[indices]
        else:
            return outcomes
    # If return_all_outcomes=True: Either return the most likely outcome (in +-1 representation),
    # or zeros are returned as outcome, if several outcomes have the same maximum probability. 
    else:
        if len(indices)==1:
            if return_probabilities:
                return outcomes, probabilities[indices]
            else:
                return outcomes
        else:
            if return_probabilities:
                return np.zeros(len(outcomes[0])), probabilities[indices[0]]
            else:
                return np.zeros(len(outcomes[0]))


def test_error_rate_quantum(n:int, m:int, runs=1000, use_QI_backend=False, api=None, num_updates=1, gate_fusion=False,
                separate_outputs=True, output_std=True, output_probabilities=True, 
                logfile_name='./log.txt',errorlog_name='./errorlog.txt',num_runs=1,verbose=False,shots=0):
    """
    Estimate average error rate, when requiring that all stored patterns must be retrieved perfectly.
    In each iteration, new stored patterns are generated uniformly at random. 
    To run the HNN, a quantum version is applied.
    The Hebbian weighting matrix is calculated and used to estimate the gate parameters of the quantum circuit.
    
    Args:   - n: input dimension
            - m: number of stored patterns
            - runs: number of iterations for average
            - separate_outputs: If true, simulation is done for each output qubit separately, 
                thus only n+1 qubits need to be simulated. Use this for local simulations.
                Else, the whole 2n-qubit circuit is simulated. Use this for Quantum Inspire, as long as n<14
            - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.

            If use_QI_backend=True:
            - api to connect to QuantumInspire
            - num_updates: number of synchronous updates, so how often the updating scheme is applied

            If use_QI_backend=False:
            - gate_fusion parameter for ProjectQ Simulator, may or may not speed up simulation

    Returns:  - Average rate of how often at least one stored pattern cannot be retrieved after num_updates updates
            - Standard deviation if output_std = True
            - runs x m-array with probabilities of all estimated most likely outcomes (only if output_probabilities=True)
    """
    at_least_one_fail = np.zeros(runs,dtype=bool)
    probabilities = -np.zeros((runs,m))

    # Repeat procedure runs times for averaging
    for k in range(runs):
        # Randomly initialize memories and weights
        W, x = new_weights(n,m)
        distances = np.zeros(m)

        # Write progress in log file 
        if verbose:
            if k % (runs//5) == 0:
                with open (logfile_name,'a') as f:
                    f.write('.')

        # Test whether all the m memories can be perfectly recovered
        for p in range(m):
            
            # Catch (connection) problems with QuantumInspire to continue computation 
            # Simply try it 16 times and wait in case of errors, usually this is quite stable
            for _ in range(16):
                try:
                    if separate_outputs:
                        updated,probabilities[k,p] = quantumHNN_update_single_outputs(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                        api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                        return_all_outcomes=False,return_probabilities=True,shots=shots)
                    else:
                        updated,probabilities[k,p] = quantumHNN_update(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                        api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                        return_all_outcomes=False,return_probabilities=True,shots=shots)
                except ErrorMessage:
                    with open (logfile_name,'a') as f:
                        f.write('({}) ErrorMessage caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                    time.sleep(10*60)
                    continue
                except Exception as e:
                    with open (logfile_name,'a') as f:
                        f.write('({}) Unexpected error caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                    with open (errorlog_name,'a') as f:
                        f.write('({}) {}\n'.format(time.strftime('%H:%M:%S'),str(e)))
                    time.sleep(10*60)
                    continue
                break

            # Test distances of updated and stored pattern, record error if things failed
            distances[p] = hamming(updated,x[p])*n
            if distances[p] > 0:
                at_least_one_fail[k] = True
                break # if a single error occurs, no need to test all remaining patterns (at least in strict formulation of errors)

    # go to next line in log file
    if verbose:
        with open (logfile_name,'a') as f:
            f.write('\n')
    # Return results according the demanded outputs
    if output_std:
        if output_probabilities:
            return np.average(at_least_one_fail), np.std(at_least_one_fail), probabilities
        else:
            return np.average(at_least_one_fail), np.std(at_least_one_fail)
    else:
        if output_probabilities:
            return np.average(at_least_one_fail), probabilities
        else:
            return np.average(at_least_one_fail)

# Specific for ProjectQ Simulator backend
def get_probabilities_from_simulator(engine,qureg,shots=0):
    """
    Returns probabilities of all computational basis states of a given register. 

    Args:
        - engine is a valid engine of ProjectQ
        - qureg is a valid list of qubits / quantum register. 
        If only one qubit, make sure to pass [qubit]!
        - number of shots. If 0 (default), the true probabilities are returned, 
            else shots = #experiments are simulated using np.random.choice based on true probabilities.

    Returns:   
        - 1d-array containing probabilities of computational basis states
        The first entry corresponds to |0>, the last to |n-1>.

    """
    try:
        n = len(qureg)
    except:
        raise TypeError('qureg must be iterable')
    
    labels = binary_labels(n)
    probabilities = np.zeros(len(labels), dtype=float)
    for i in range(len(labels)):
        probabilities[i] = engine.backend.get_probability(labels[i],qureg)
    
    if shots==0:
        return probabilities / np.sum(probabilities)
    else:
        # based on true probabilities, employ RNG to simulate outcomes.
        true_probabilities = probabilities / np.sum(probabilities)

        # Simulate real experiments, draw a sample accoring to theoretical probability distribution
        indices = np.arange(len(labels))
        experiments = np.random.choice(indices,size=shots,p=true_probabilities)

        # count outcomes
        occurances = np.zeros(len(labels),dtype=int)
        for i in experiments:
            occurances[i] += 1

        return occurances/np.sum(occurances)



#%% 
# Optimized implementation, where a single output is considered only 
def U_ZX_circuit_single_output(engine,input_register,output_qubit,i,weighting_matrix):
    """ 
    Applies circuit to quantum register consisting of all unitaries e^(i theta Z_jxX_i) for all j and a given input qubit,
    excluding self-connections, so i!=j.

    Args:
        - engine is a valid engine of ProjectQ
        - input_register is a non-empty quantum register
        - output_qubit is a single output qubit
        - i is the index of the output_qubit within the output register
        - weighting matrix is the one from Hebbian learning, controlling the gate parameters
        
    """
    n_qubits = len(input_register)
    assert(n_qubits>1), 'n_bits is too small, dividing by (n-1) causes troubles'
    
    # Apply H to all output-qubits before and after CNOTs/rotations, the H in the middle of circuit cancle each other out
    H | output_qubit

    # Factor for all angles, multiply with weighting_matrix[i,j] 
    a = np.pi/4 /(n_qubits-1)

    # Apply all gates and discard self-connection 
    # From 0 until i-1
    for j in range(i):
        theta = a * weighting_matrix[i,j]
        CNOT | (input_register[j],output_qubit)
        Rz(2*theta) | output_qubit
        CNOT | (input_register[j],output_qubit)
    # From i+1 onwards 
    for j in np.arange(i+1,n_qubits):
        theta = a * weighting_matrix[i,j]
        CNOT | (input_register[j],output_qubit)
        Rz(2*theta) | output_qubit
        CNOT | (input_register[j],output_qubit)
    
    # Apply H to all output-qubits before and after CNOTs/rotations, the H in the middle of circuit cancle each other out
    H | output_qubit

def quantumHNN_update_single_outputs(W:np.ndarray, query:np.ndarray, use_QI_backend=False, api= None, gate_fusion=False, verbose=False, 
                    num_runs=1, return_all_outcomes=False, return_probabilities=False,shots=0):
    """
    Sets up and evaluates quantum circuit of feed-forward like implementation of HNN 
    for a given weighting matrix and query. 
    Contrary to quantumHNN_update, here each output qubit is considered separately and all results are combined in the end.
    The simulation is performed either on QuantumInspire (externally) or ProjectQ (locally) simulators

    Args:
        - W is a 2d-array as weighting matrix
        - query is a 1d-array with entries +-1
        - if use_QI_backend = True, QuantumInspire is used, default is the ProjectQ Simulator
        - api to connect to QuantumInspire
        - num_runs is the parameter of QuantumInspire
        - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.
    
    Returns:
        - most likely outcomes (If return_all_outcomes=True, all outcomes with maximum probability are returned.
        Else, only one outcome is returned. In the latter case, a zero-array is returned if multiple outcomes have same probability.)
        - If return_probabilities=True, also the according probabilities are returned
    """
    # Check inputs
    assert(query.ndim == 1), 'Query array is of wrong shape'
    assert(np.all(query**2)), 'Query contains entries other than +-1'
    assert(W.ndim == 2), 'W is of wrong shape'
    n_bits = len(query)
    assert(np.shape(W)[0] == n_bits), 'Sizes of W and query do not match'

    # Store the most likely outcomes in this array
    single_probabilities_combined = np.zeros((n_bits,2))
    
    # Loop over all output qubits
    for i in range(n_bits):
        # Allocate quantum register
        eng = create_engine(use_QI_backend=use_QI_backend, api=api, gate_fusion= gate_fusion, num_runs=num_runs,verbose=verbose)
        input_register = eng.allocate_qureg(n_bits)
        output_qubit = eng.allocate_qubit()

        # Initialize input register according query sting
        state_preparation(input_register,query)

        # Initialize all output qubits as |+i>
        Rx(-np.pi/2) | output_qubit

        # Run actual ciruit
        U_ZX_circuit_single_output(eng,input_register,output_qubit,i,W)

        # Flush whole circuit down the pipeline     
        eng.flush()

        # Extract outcome probabilities of Y-measurement of circuit
        if use_QI_backend:
            output_dict = eng.backend.get_probabilities(output_qubit)
            single_probabilities = probabilities_from_dict(output_dict)
        else: 
            single_probabilities = get_probabilities_from_simulator(eng,output_qubit,shots=shots)

            # Make sure to be in computational basis state before deallocating
            All(Measure) | input_register
            Measure | output_qubit
        
        # Store probabilities of outcomes when measuring the output qubit
        single_probabilities_combined[i] = single_probabilities
    
    # get probabilities of strings rather than the single bits by multiplying the corresponding probabilities
    probabilities = single_outcomes_to_register_outcome(single_probabilities_combined)
    outcome,indices = most_likely_outcome(probabilities, output_index=True)  

    # If return_all_outcomes=True: Return all outcomes, no matter whether a single or multiple outcomes possible
    # If return_probabilities=True, also the according probabilities are returned
    if return_all_outcomes:
        if return_probabilities:
            return outcome, probabilities[indices]
        else:
            return outcome
    # If return_all_outcomes=True: Either return the most likely outcome (in +-1 representation),
    # or zeros are returned as outcome, if several outcomes have the same maximum probability. 
    else:
        if len(indices)==1:
            if return_probabilities:
                return outcome, probabilities[indices]
            else:
                return outcome
        else:
            if return_probabilities:
                return np.zeros(len(outcome[0])), probabilities[indices[0]]
            else:
                return np.zeros(len(outcome[0]))


#%% Functions for parallelization
#%%
def parallel_method(n, m, separate_outputs, verbose, api,logfile_name, errorlog_name,
                        use_QI_backend, gate_fusion, num_runs,new_weights,shots=0):
    """
    Parallelize the averaging part of test_error_rate_quantum. 
    This function shoud only be used within the Pool in test_error_rate_quantum_parallel!

    - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.
    """
    # When using multiple processes at the same time, np.random seeds itself based on some OS-variable 
    # or wallclock time and it is likely that the same seed occurs in many processes. Thus explicitly call a new seed
    # best would actually be to generate seeds for each process beforehand and pass them to here 
    np.random.seed()

    # Randomly initialize memories and weights
    W, x = new_weights(n,m)
    
    distances = np.zeros(m)
    probabilities = np.zeros(m,dtype=float)
    at_least_one_fail = False

    # Test whether all the m memories can be perfectly recovered
    for p in range(m):
        
        # Catch (connection) problems with QuantumInspire to continue computation 
        # Simply try it 16 times and wait in case of errors, usually this is quite stable
        for _ in range(16):
            try:
                if separate_outputs:
                    updated,probabilities[p] = quantumHNN_update_single_outputs(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                    api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                    return_all_outcomes=False,return_probabilities=True,shots=shots)
                else:
                    updated,probabilities[p] = quantumHNN_update(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                    api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                    return_all_outcomes=False,return_probabilities=True,shots=shots)
            except ErrorMessage:
                with open (logfile_name,'a') as f:
                    f.write('({}) ErrorMessage caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                time.sleep(10*60)
                continue
            except Exception as e:
                with open (logfile_name,'a') as f:
                    f.write('({}) Unexpected error caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                with open (errorlog_name,'a') as f:
                    f.write('({}) {}\n'.format(time.strftime('%H:%M:%S'),str(e)))
                time.sleep(10*60)
                continue
            break

        # Test distances of updated and stored pattern, record error if things failed
        distances[p] = hamming(updated,x[p])*n
        if distances[p] > 0:
            at_least_one_fail = True
            break # if a single error occurs, no need to test all remaining patterns (at least in strict formulation of errors)
    
    return at_least_one_fail, probabilities

def test_error_rate_quantum_parallel(n:int, m:int, runs=1000, use_QI_backend=False, api=None, num_updates=1, gate_fusion=False,
                separate_outputs=True, output_std=True, output_probabilities=True, processes = None,
                logfile_name='./log.txt', errorlog_name='./errorlog.txt', num_runs=1, verbose=False, shots=0):
    """
    Estimate average error rate, when requiring that all stored patterns must be retrieved perfectly.
    In a parallelized fashion, new stored patterns are generated uniformly at random. 
    To run the HNN, a quantum version is applied.
    The Hebbian weighting matrix is calculated and used to estimate the gate parameters of the quantum circuit.
    
    Note: Do not use this function if OS is Windows! (Typically causes problems) 

    Args:   - n: input dimension
            - m: number of stored patterns
            - runs: number of iterations for average
            - separate_outputs: If true, simulation is done for each output qubit separately, 
                thus only n+1 qubits need to be simulated. Use this for local simulations.
                Else, the whole 2n-qubit circuit is simulated. Use this for Quantum Inspire, as long as n<14

            If use_QI_backend=True:
            - api to connect to QuantumInspire
            - num_updates: number of synchronous updates, so how often the updating scheme is applied

            If use_QI_backend=False:
            - gate_fusion parameter for ProjectQ Simulator, may or may not speed up simulation
            - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.

    Returns:  - Average rate of how often at least one stored pattern cannot be retrieved after num_updates updates
            - Standard deviation if output_std = True
            - runs x m-array with probabilities of all estimated most likely outcomes (only if output_probabilities=True)
    """
    # Seems to not work under Windows, thus abort then
    #%%
    assert(platform != 'win32'), 'Parallelization with multiprocess package does not work with Windows'
    
    #at_least_one_fail = np.zeros(runs,dtype=bool)
    probabilities = -np.zeros((runs,m))

    # Repeat procedure runs times for averaging in a parallelized fashion
    pool = Pool(processes= processes)
    results = [pool.apply_async(parallel_method,args=(n,m,separate_outputs,verbose,api,logfile_name,errorlog_name,
                                            use_QI_backend, gate_fusion, num_runs,new_weights,shots)) for _ in range(runs)]
    output = [process.get() for process in results]
    output= np.array(output)
    
    # Format the outputs and store in proper arrays
    at_least_one_fail = np.array(output[:,0],dtype=bool)
    for k in range(runs):
        for p in range(m):
            probabilities[k,p] =  output[k,1][p]
    
    # go to next line in log file
    if verbose:
        with open (logfile_name,'a') as f:
            f.write('\n')
    # Return results according the demanded outputs
    if output_std:
        if output_probabilities:
            return np.average(at_least_one_fail), np.std(at_least_one_fail), probabilities
        else:
            return np.average(at_least_one_fail), np.std(at_least_one_fail)
    else:
        if output_probabilities:
            return np.average(at_least_one_fail), probabilities
        else:
            return np.average(at_least_one_fail)



# Functions for all definitions of errors
#%%
def parallel_method_attraction(n, m, distinct_memories, separate_outputs, verbose, api,logfile_name, errorlog_name,
                        use_QI_backend, gate_fusion, num_runs,new_weights,shots=0):
    """
    Parallelize the averaging part of test_error_rate_quantum. 
    This function shoud only be used within the Pool in test_error_rate_quantum_parallel!

    - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.
    """
    # When using multiple processes at the same time, np.random seeds itself based on some OS-variable 
    # or wallclock time and it is likely that the same seed occurs in many processes. Thus explicitly call a new seed
    # best would actually be to generate seeds for each process beforehand and pass them to here 
    np.random.seed()

    # Randomly initialize memories and weights
    W, x = new_weights(n,m,distinct_memories=distinct_memories)
    
    distances = np.zeros(m)
    probabilities = np.zeros(m,dtype=float)
    at_least_one_fail = False

    # Test whether all the m memories can be perfectly recovered
    for p in range(m):
        
        # Catch (connection) problems with QuantumInspire to continue computation 
        # Simply try it 16 times and wait in case of errors, usually this is quite stable
        for _ in range(16):
            try:
                if separate_outputs:
                    updated,probabilities[p] = quantumHNN_update_single_outputs(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                    api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                    return_all_outcomes=False,return_probabilities=True,shots=shots)
                else:
                    updated,probabilities[p] = quantumHNN_update(W/m,x[p],use_QI_backend=use_QI_backend, 
                                                    api=api, gate_fusion=gate_fusion,num_runs=num_runs,
                                                    return_all_outcomes=False,return_probabilities=True,shots=shots)
            except ErrorMessage:
                with open (logfile_name,'a') as f:
                    f.write('({}) ErrorMessage caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                time.sleep(10*60)
                continue
            except Exception as e:
                with open (logfile_name,'a') as f:
                    f.write('({}) Unexpected error caught\n'.format(time.strftime('%d.%m.%y, %H:%M:%S')))
                with open (errorlog_name,'a') as f:
                    f.write('({}) {}\n'.format(time.strftime('%H:%M:%S'),str(e)))
                time.sleep(10*60)
                continue
            break

        # Test distances of updated and stored pattern, record error if things failed
        distances[p] = hamming(updated,x[p])*n

    # Estimate the corresponding error rates
    BitErrorRate_m = np.mean(distances/n)
    MessageErrorRate_m = np.mean(distances>0)                 

    return BitErrorRate_m, MessageErrorRate_m, probabilities

def test_error_rate_quantum_parallel_attraction(n:int, m:int, runs=1000, use_QI_backend=False, api=None, num_updates=1, gate_fusion=False,
                separate_outputs=True, output_std=True, output_probabilities=True, processes = None, distinct_memories=True,
                logfile_name='./log.txt', errorlog_name='./errorlog.txt', num_runs=1, verbose=False, shots=0):
    """
    Estimate average SER, MER and BER.
    In a parallelized fashion, new stored patterns are generated uniformly at random. 
    To run the HNN, a quantum version is applied.
    The Hebbian weighting matrix is calculated and used to estimate the gate parameters of the quantum circuit.
    
    Note: Do not use this function if OS is Windows! (Typically causes problems) 

    Args:   - n: input dimension
            - m: number of stored patterns
            - runs: number of iterations for average
            - separate_outputs: If true, simulation is done for each output qubit separately, 
                thus only n+1 qubits need to be simulated. Use this for local simulations.
                Else, the whole 2n-qubit circuit is simulated. Use this for Quantum Inspire, as long as n<14

            If use_QI_backend=True:
            - api to connect to QuantumInspire
            - num_updates: number of synchronous updates, so how often the updating scheme is applied

            If use_QI_backend=False:
            - gate_fusion parameter for ProjectQ Simulator, may or may not speed up simulation
            - shots = number of experiments. If 0, true probabilities are considered, 
            else experiments are simulated using np.random.choice, based on true probabilities.

    Returns:  - SER, SER_std, MER, MER_std, BER, BER_std
            - runs x m-array with probabilities of all estimated most likely outcomes (only if output_probabilities=True)
    """
    # Seems to not work under Windows, thus abort then
    #%%
    # assert(platform != 'win32'), 'Parallelization with multiprocess package does not work with Windows'
    
    #at_least_one_fail = np.zeros(runs,dtype=bool)
    probabilities = -np.zeros((runs,m))

    # Repeat procedure runs times for averaging in a parallelized fashion
    pool = Pool(processes= processes)
    results = [pool.apply_async(parallel_method_attraction,args=(n,m,distinct_memories,separate_outputs,verbose,api,logfile_name,errorlog_name,
                                            use_QI_backend, gate_fusion, num_runs,new_weights,shots)) for _ in range(runs)]
    output = [process.get() for process in results]
    output= np.array(output)
    
    # Format the outputs and store in proper arrays
    BER = np.array(output[:,0])
    MER = np.array(output[:,1])
    SER = np.array(output[:,1]>0)
   

    for k in range(runs):
        for p in range(m):
            probabilities[k,p] =  output[k,2][p]
    
    # go to next line in log file
    if verbose:
        with open (logfile_name,'a') as f:
            f.write('\n')
    # Return results according the demanded outputs
    if output_probabilities:
        return np.average(SER), np.std(SER), np.average(MER), np.std(MER), np.average(BER), np.std(BER), probabilities
    else:
        return np.average(SER), np.std(SER), np.average(MER), np.std(MER), np.average(BER), np.std(BER)

