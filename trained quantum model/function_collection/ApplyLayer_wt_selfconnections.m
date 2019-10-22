%Application of all unitaries in one given layer. This corresponds to the
%action of the \mathcal{E} on a given start density matrix rho_in. Note
%that self-connections are removed, so the unitary U_j acts on all input
%qubits other than the j-th and the j-th output qubit
function rho_out = ApplyLayer_wt_selfconnections(rho_in,U,n)

%pho_in (pho_out): input (output) state of the corresponding layer
%n: Number of neurons in both input and output layer
%U array containing all n unitaries, acting on all inputs (except for the
%j-th one) and the j-th output qubit


%Initialising Input State on all 2n qubits
RHO_in_whole = kron(rho_in,[1;zeros(2^n -1,1)]*[1;zeros(2^n -1,1)]');

%Calculating Output on all N_CorrLay + N_PrevLay Qubits
RHO_out_whole = RHO_in_whole;
for j= 1:n
    % First: tensor U_j with identiy on n additional quits
    % Then: swap j-th and n+j-th qubit, st. identiy acts on j-th input 
    % qubit, while U_j acts non-trivially on j-th output qubit
    V = Swap(kron(U(:,:,j),eye(2^n)),[j,n+j],2*ones(1,2*n));
    RHO_out_whole = V*RHO_out_whole*V';
end

% Calculating Output state of the neurons in the output layer by tracing
% out all inpts
rho_out = PartialTrace(RHO_out_whole,1:n,2*ones(1,2*n));



end