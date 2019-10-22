% Calculates Cost function of Network given a set of input and target output
% states.
% Note that self-connections are excluded, meaning that U acts on all input
% qubits except for the j-th, and on the j-th output qubit
function C = CostNetwork_wt_selfconnections(phi_in,phi_out,U,n)

%phi_in (phi_out): array with columns being the input (output) vectors
N_NumTrain = size(phi_in,2);
C = 0;

for x=1:N_NumTrain  
    rho_in_x = phi_in(:,x)*phi_in(:,x)';
    C = C + dot(phi_out(:,x),ApplyLayer_wt_selfconnections(rho_in_x,U{2},n)*phi_out(:,x));
end

C = real(C/N_NumTrain);

end