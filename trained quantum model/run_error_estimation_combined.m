addpath('./DeepQNN-master/DeepQNN-MATLAB')
addpath('./function_collection')

N_workers = 4;
delete(gcp('nocreate'))
pool = parpool('local', N_workers);

% Main parameters
n = 3;
m_start = 1;
m_stop = 8;
rounds = 500;
training_iterations = 50; 
distance = 1;
lambda = 1;
distinct = true;
selfconnections = true;
N_vicinity = 100;

%wt_selfconnections/
logfile_name = sprintf('./results/combined_rates/19_09_30_n%d_rounds%d_d%d_iter%d_lambda%d_distinct_wtSelfconnections_log.txt',n,rounds,distance,training_iterations,lambda);
save_data_path = sprintf('./results/combined_rates/19_09_30_n%d_rounds%d_d%d_iter%d_lambda%d_distinct_wtSelfconnections.mat',n,rounds,distance,training_iterations,lambda);

fileID = fopen(logfile_name,'a');
fprintf(fileID,'Use pool with %d workers\n',N_workers);
fclose(fileID);

% wt_selfconnections, ,distinct
[~,~,~,~] = ErrorRateEstimation_combined(n, m_start, m_stop,N_vicinity, distance, rounds, training_iterations, lambda, logfile_name,save_data_path,distinct,selfconnections);

pool.delete()
