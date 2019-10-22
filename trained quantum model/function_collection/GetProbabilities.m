% Computes probabilities of outcomes for a given quantum state rho as 
% vector prob, which is already ordered: first entry corresponds to |0> 
% and last to |N-1>
function prob = GetProbabilities(rho)

% Born rule reduces to the diagonal of the density matrix, since
% computational basis considered
prob = real(diag(rho)); 

end 