% Returns the mean error rates (MER) and their standard deviations for all m 
% within 1 and the given dimension n as (n,1) vectors
% If include_BER==True: also calculate these
function [MER_rates,BER_rates,MER_rates_std,BER_rates_std] = ErrorRateEstimation_combined(n, m_start, m_stop, N_vicinity, distance, rounds, training_iterations, lambda, logfile_name,save_data_path,distinct,selfconnections)

% Initialize data arrays
m = m_start:m_stop;
% perfect retrieval with strict definition
error_rates_strict = -1* ones(n,1);
std_rates_strict = -1* ones(n,1);

% MER(=error_rates) and BER for non-noisy inputs
error_rates = -1* ones(n,1);
std_rates = -1* ones(n,1);
BER = -1* ones(n,1);
std_BER = -1* ones(n,1);
CostTraining_all = -1*ones(m_start-m_stop+1,rounds,training_iterations);
CostMemories = -1*ones(m_start-m_stop+1,rounds,m_stop);

% MER(=error_rates) and BER for noisy inputs
MER_rates = -1* ones(3,n);
MER_rates_std = -1* ones(3,n);
BER_rates = -1* ones(3,n);
BER_rates_std = -1* ones(3,n);

% Write settings into log file
fileID = fopen(logfile_name,'a');
fprintf(fileID,'Estimate error rate for n= %d\n',n);
if distinct==true
    fprintf(fileID,'Memories must be distinct\n');
else
    fprintf(fileID,'Memories are not necessarily distinct\n');
end
if selfconnections==true
    fprintf(fileID,'Self-connections are removed\n');
else
    fprintf(fileID,'Network has self-connections\n');
end
fprintf(fileID,'Use test implementation without storing output unitaries each time\n');
fprintf(fileID,'Combine all error rate estimations to reduce runtime\n');
fprintf(fileID,'rounds = %d\ndistance = %d\ntraining iterations = %d\nlambda = %.3f\nNvic = %d\n\v',rounds, distance,training_iterations,lambda,N_vicinity);
fprintf(fileID,'m_Start = %d\nm_end = %d\n\v',m_start, m_stop);
fclose(fileID);

tic

for i = m
    % Write progress into log file
    fileID = fopen(logfile_name,'a');
    fprintf(fileID,'%d / %d (%s)\n',i,n,datestr(now));
    fclose(fileID);
    [error_rate_m_strict, std_error_rate_m_strict,error_rate_m, std_error_rate_m,BER_m,std_BER_m,CostTraining_all_m,CostMemories_m, MER_mean, BER_mean,MER_std,BER_std] = ErrorRateFixedMemory_combined(n,i,N_vicinity,distance,rounds,training_iterations,lambda,logfile_name,distinct,selfconnections);
    
    % strict error rates
    error_rates_strict(i) = error_rate_m_strict;
    std_rates_strict(i) = std_error_rate_m_strict;
    
    % error rates for memories as inputs
    BER(i) = BER_m;
    std_BER(i) = std_BER_m;
    error_rates(i) = error_rate_m;
    std_rates(i) = std_error_rate_m;
    CostTraining_all(i,:,:) = CostTraining_all_m;
    CostMemories(i,:,1:i) = CostMemories_m;
    
    % error rates for noisy inputs
    MER_rates(:,i) = MER_mean;
    MER_rates_std(:,i) = MER_std;
    BER_rates(:,i) = BER_mean;
    BER_rates_std(:,i) = BER_std;
    
    save(save_data_path,'m','MER_rates','BER_rates','MER_rates_std','BER_rates_std','error_rates_strict','std_rates_strict','error_rates','std_rates','CostTraining_all','CostMemories','BER','std_BER')
    
    % timeElapsed1 = toc;
    % fprintf('Duration = %.3f\n',timeElapsed1);
end

timeElapsed = toc;
% fprintf('Duration = %.3f\n',timeElapsed);

fileID = fopen(logfile_name,'a');
fprintf(fileID,'\vFinished on %s\nTotal duration: %s\n',datestr(now),datestr(seconds(timeElapsed),'HH:MM:SS'));
fprintf(fileID,'error_rates_strict = %s \n',mat2str(round(error_rates_strict.',2)));
fprintf(fileID,'std_rates_strict = %s \n \n',mat2str(round(std_rates_strict.',2)));
fprintf(fileID,'error_rates = %s \n',mat2str(round(error_rates.',2)));
fprintf(fileID,'std_rates = %s \n \n',mat2str(round(std_rates.',2)));
fprintf(fileID,'MER_rates = %s \n',mat2str(round(MER_rates.',2)));
fprintf(fileID,'MER_rates_std = %s \n',mat2str(round(MER_rates_std.',2)));
fprintf(fileID,'BER_rates = %s \n',mat2str(round(BER_rates.',2)));
fprintf(fileID,'BER_rates_std = %s \n',mat2str(round(BER_rates_std.',2)));
fclose(fileID);

end