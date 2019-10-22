% Calculate fidelity/cost function of the actual outcome as density matrix 
% with respect to the desired output
function C = Fidelity(actual_output, desired_output)

C = dot(desired_output,actual_output*desired_output);
C = real(C);

end 