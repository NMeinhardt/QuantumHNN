%Calculates the output state/density matrix for some given input quantum
%state described by density matrix rho_in
function rho_out = ApplyNetworkQuantum(rho_in,U,M)

% M array of Number of Neurons in each layer, i.e. size(M,2) = Num_Layers, M(1,j) = Num_Neurons in Layer j


N_Layers = size(M,2);

rho_out = rho_in;

for k = 2:N_Layers
   rho_out = ApplyLayer(rho_out,U{k},M(k),M(k-1));
end


end