
function U = QuickInitilizer_wt_selfconnections(n)

%This functions quickly initilizes a set of random unitaries for the
%network with 2 layers, both of length n
%The unitaries for each layer are stored in the cell array U.
%U{k}(:,:,j) is the unitary acting on the jth neuron in the kth layer.
%Self-connections are removed, such that U{2}(:,:,j) acts on 2^n qubits,
%which are the n-1 input qubits other than the j-th one and the j-th
%ancilla qubit

for j = 1:n
    U{2}(:,:,j) = Randomunitary(2^n);
        
end

