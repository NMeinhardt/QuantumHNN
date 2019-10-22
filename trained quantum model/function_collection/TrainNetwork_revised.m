%Trains the Network and gives out the array with all trained unitaries in U
%and an array CList with all Cost functions while training
function [U,CList] = TrainNetwork_revised(phi_in,phi_out,U,n,lambda,iter)

%iter: number of iterations the network trains (update all unitaries)
eps= 0.1;

N_NumTrain = size(phi_in,2);
M = [n,n];

% Calculate initial cost
CList = [CostNetwork(phi_in,phi_out,U,M)];



%Update the Unitaries iter times

for round=2:iter

    %Generating all K Update Matrices
    for x = 1:N_NumTrain
        
        %Initilize a state to calculate state of left side of the Commutator in the Update Matrix M
        %i.e. the one coming from the \mathcal{E} or "ApplyLayer" Channel.
        if x == 1
           K = zeros(2^(n+1),2^(n+1),n);
        end

        rho_left_prev = phi_in(:,x)*phi_in(:,x)';

        rho_left =  kron(rho_left_prev,[1;zeros(2^n-1,1)]*[1;zeros(2^n-1,1)]');
        
      for j = 1:n
          %Initilize a state to calculate the state of the right hand side
          %of the Commutator in the Update Matrix M, i.e. the one coming
          %from the conjugate F Channel.
          if j==1 
             rho_right_prev = phi_out(:,x)*phi_out(:,x)';
             rho_right = kron(eye(2^n),rho_right_prev);
             for j_1 = 1:n
                 j_2 = n -j_1 +1;
                 V = Swap(kron(U{2}(:,:,j_2),eye(2^(n-1))),[n+1,n+j_2],2*ones(1,n+n));
                rho_right = V'*rho_right*V;
             end
          end
               
      %Generating left hand side of commutator for M_j^k. Note that we can use application
      %of all unitaries before the _j^k Neuron
               V = Swap(kron(U{2}(:,:,j),eye(2^(n-1))),[n+1,n+j],2*ones(1,n+n));
               rho_left = V*rho_left*V';
              
               rho_right = V*rho_right*V';
               
               M_Update = Comm(rho_left,rho_right);
                            
               K(:,:,j) = K(:,:,j) + PartialTrace(M_Update,[n+1:n+j-1,n+j+1:n+n] , 2*ones(1,n+n)); 

      end
      
    end
     
    

%Updating all Unitaries in the Network

for j = 1:n
    U{2}(:,:,j) =expm((-eps*2^n/(N_NumTrain*lambda))*K(:,:,j))*U{2}(:,:,j);
end


%Save the Costfunction of this round
CList(round) = CostNetwork(phi_in,phi_out,U,M);

end

end
