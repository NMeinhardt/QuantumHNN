% Returns all errror rates to save runtime, as training is most costful
function [error_rate_strict,std_error_rate_strict,error_rate, std_error_rate,BER_m,std_BER_m,CostTraining_all,CostMemories,MER_mean,BER_mean,MER_std,BER_std] = ErrorRateFixedMemory_combined(n, m, N_vicinity, distance, rounds, training_iterations, lambda, logfile_name, distinct,selfconnections)


% Read in training set
if distinct == true
    filename = sprintf('./data_vicinity/trainingset_n%d_rounds%d_Nvic%d_m%d_d%d_distinct.mat',n,rounds,N_vicinity,m,distance);
else
    filename = sprintf('./data_vicinity/trainingset_n%d_rounds%d_Nvic%d_m%d_d%d.mat',n,rounds,N_vicinity,m,distance);
end
load(filename,'phi_in','phi_out','memory_states','memory_int_representation');

% import test set depending on noise rate:
load(filename,'test_states_noise10','test_states_noise20','test_states_noise30');

test_states10 = test_states_noise10;
test_states20 = test_states_noise20;
test_states30 = test_states_noise30;


% initialize arrays to store error rates for noisy inputs
MER_10 = -1* ones(rounds,1);
MER_20 = -1* ones(rounds,1);
MER_30 = -1* ones(rounds,1);
BER_10 = -1* ones(rounds,1);
BER_20 = -1* ones(rounds,1);
BER_30 = -1* ones(rounds,1);

% initialize arrays to store data for memories as inputs
error_rate_rounds = -1* ones(rounds,1);
bit_error_rate_rounds = -1* ones(rounds,1);
CostTraining_all = -1* ones(rounds,training_iterations);
CostMemories = -1 * ones(rounds,m); 

% Initialize error rates for strict definition
error_rate_rounds_strict = -1* zeros(rounds,1);

parfor r = 1:rounds
    
    % Write progress into log file
%     fileID = fopen(logfile_name,'a');
%     fprintf(fileID,'.');
%     fclose(fileID);
    % fprintf('Start round  %d\n',r);
    
    % choose data of this round
    phi_in_round = squeeze(phi_in(r,:,:));
    phi_out_round = squeeze(phi_out(r,:,:));
    % don't use squeeze for memory_states_round, as else m=1 results in
    % wrong shape of array
    memory_states_round = reshape(memory_states(r,:,:),[2^n,m]); 
    memory_int_representation_round = squeeze(memory_int_representation(r,:,:));
    test_states10_round = reshape(test_states10(r,:,:),[2^n,m*N_vicinity]); 
    test_states20_round = reshape(test_states20(r,:,:),[2^n,m*N_vicinity]); 
    test_states30_round = reshape(test_states30(r,:,:),[2^n,m*N_vicinity]); 
    
    if selfconnections
        % Initialize unitaries
        U_ini = QuickInitilizer_wt_selfconnections(n);

        % Run training
        [U_final,CostTraining] = TrainNetwork_wt_selfconnections(phi_in_round,phi_out_round,U_ini,n,lambda,training_iterations);
    else
        % Initialize unitaries
        U_ini = QuickInitilizer([n,n]);

        % Run training
        [U_final,CostTraining] = TrainNetwork_revised(phi_in_round,phi_out_round,U_ini,n,lambda,training_iterations);
    end 
    
    % Store parameters of current training 
    CostTraining_all(r,:) = CostTraining;
    
    % Set error counters to zero for presenting memories
    errors = 0;
    bit_errors = 0;
    
    % Set error counters to zero 
    message_errors_10 = 0;
    message_errors_20 = 0;
    message_errors_30 = 0;
    bit_errors_10 = 0;
    bit_errors_20 = 0;
    bit_errors_30 = 0;
    
    % Test whether memories themselves can be retrieved, so whether they are
    % stable states
    for memory = 1:m
        % estimate MER, BER and strict error rate for memories themselves
        if selfconnections
            rho_in_m = memory_states_round(:,memory)*memory_states_round(:,memory)';
            rho_out_m = ApplyLayer_wt_selfconnections(rho_in_m,U_final{2},n);
        else
            rho_out_m = ApplyNetwork(memory_states_round(:,memory),U_final,[n,n]);
        end
        prob = GetProbabilities(rho_out_m);
        C = Fidelity(rho_out_m, memory_states_round(:,memory))
        CostMemories(r,memory) = C;
        
        % Get the most likely outcome, its integer representation is the argmax of
        % the probability array minus 1 (since Matlab starts counting at 1)
        [index, ~] = MostLikelyOutcome(prob);

        % If the most likely outcome does not correspond to the stored memory
        % of if several outcomes are equally likely, assign an error
        if length(index) > 1 || memory_int_representation_round(memory)~= index -1
            errors = errors+1;
        end
        
        % test bit errors
        bin_rep_memory = decimalToBinaryVector(memory_int_representation_round(memory),n);
        bin_rep_outcome = decimalToBinaryVector(index -1,n);
        bit_errors = bit_errors + pdist2(bin_rep_memory,bin_rep_outcome,'hamming')*n;
        
        % Estimate error rates for noisy samples
        for j = 1:N_vicinity
            % Run circuit with U_final for given test pattern as input and get
            % probabilities of comp basis states  
            if selfconnections
                rho_in_m_10 = test_states10_round(:,(memory-1)*N_vicinity+j)*test_states10_round(:,(memory-1)*N_vicinity+j)';
                rho_out_m_10 = ApplyLayer_wt_selfconnections(rho_in_m_10,U_final{2},n);
                rho_in_m_20 = test_states20_round(:,(memory-1)*N_vicinity+j)*test_states20_round(:,(memory-1)*N_vicinity+j)';
                rho_out_m_20 = ApplyLayer_wt_selfconnections(rho_in_m_20,U_final{2},n);
                rho_in_m_30 = test_states30_round(:,(memory-1)*N_vicinity+j)*test_states30_round(:,(memory-1)*N_vicinity+j)';
                rho_out_m_30 = ApplyLayer_wt_selfconnections(rho_in_m_30,U_final{2},n);
            
            else
                rho_out_m_10 = ApplyNetwork(test_states10_round(:,(memory-1)*N_vicinity+j),U_final,[n,n]);
                rho_out_m_20 = ApplyNetwork(test_states20_round(:,(memory-1)*N_vicinity+j),U_final,[n,n]);
                rho_out_m_30 = ApplyNetwork(test_states30_round(:,(memory-1)*N_vicinity+j),U_final,[n,n]);
            end
            
            prob_10 = GetProbabilities(rho_out_m_10);
            prob_20 = GetProbabilities(rho_out_m_20);
            prob_30 = GetProbabilities(rho_out_m_30);
        
            % Get the most likely outcome, its integer representation is the argmax of
            % the probability array minus 1 (since Matlab starts counting at 1)
            [index_10, ~ ] = MostLikelyOutcome(prob_10);
            [index_20, ~ ] = MostLikelyOutcome(prob_20);
            [index_30, ~ ] = MostLikelyOutcome(prob_30);

            % If the most likely outcome does not correspond to the stored memory
            % of if several outcomes are equally likely, assign an error
            if length(index_10) > 1 || memory_int_representation_round(memory)~= index_10 -1
                message_errors_10 = message_errors_10 + 1;
            end
            if length(index_20) > 1 || memory_int_representation_round(memory)~= index_20 -1
                message_errors_20 = message_errors_20 + 1;
            end
            if length(index_30) > 1 || memory_int_representation_round(memory)~= index_30 -1
                message_errors_30 = message_errors_30 + 1;
            end
            
            % test bit errors
            bin_rep_memory = decimalToBinaryVector(memory_int_representation_round(memory),n);
            bin_rep_outcome_10 = decimalToBinaryVector(index_10 -1,n);
            bin_rep_outcome_20 = decimalToBinaryVector(index_20 -1,n);
            bin_rep_outcome_30 = decimalToBinaryVector(index_30 -1,n);
            bit_errors_10 = bit_errors_10 + pdist2(bin_rep_memory,bin_rep_outcome_10,'hamming')*n;
            bit_errors_20 = bit_errors_20 + pdist2(bin_rep_memory,bin_rep_outcome_20,'hamming')*n;
            bit_errors_30 = bit_errors_30 + pdist2(bin_rep_memory,bin_rep_outcome_30,'hamming')*n;
            
        end
    end
    
     % Estimate strict error rate for this round
    if errors > 0
        error_rate_rounds_strict(r) = 1;
    end
    
    % Estimate MER and BER for memories as inputs for this round
    error_rate_rounds(r) = errors/m;
    bit_error_rate_rounds(r) =bit_errors/m/n;
        
    % Estimate message error rate for this round
    MER_10(r) = message_errors_10/m/N_vicinity;
    MER_20(r) = message_errors_20/m/N_vicinity;
    MER_30(r) = message_errors_30/m/N_vicinity;
    % Estimate bit error rate for this round
    BER_10(r) = bit_errors_10/m/n/N_vicinity;
    BER_20(r) = bit_errors_20/m/n/N_vicinity;
    BER_30(r) = bit_errors_30/m/n/N_vicinity;


end

% Estimate mean and standard deviation of MER and BER for noisy inputs
MER_mean = [mean(MER_10),mean(MER_20),mean(MER_30)];
BER_mean = [mean(BER_10),mean(BER_20),mean(BER_30)];
MER_std = [std(MER_10),std(MER_20),std(MER_30)];
BER_std = [std(BER_10),std(BER_20),std(BER_30)];

% Strict error rates
error_rate_strict = mean(error_rate_rounds_strict);
std_error_rate_strict = std(error_rate_rounds_strict);

% Estimate mean and standard deviation of error rates, memories as input
error_rate = mean(error_rate_rounds);
std_error_rate = std(error_rate_rounds);
BER_m = mean(bit_error_rate_rounds);
std_BER_m = std(bit_error_rate_rounds);


% Write progress into log file
fileID = fopen(logfile_name,'a');
fprintf(fileID,'\n');
fclose(fileID);

end 