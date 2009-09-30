% Explicit examples on how to use distributions

leng=50;
rep=5;
weight=1;
order=3;
gap=0;
num=12;
len=23;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');


% Histogram
disp('Histogram');

%sg('new_distribution', 'HISTOGRAM');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');

%	sg('train_distribution');
%	histo=sg('get_histogram');

%	num_param=sg('get_histogram_num_model_parameters');
%	for i = 1:num,
%		for j = 1:num_param,
%			sg(sprintf('get_log_derivative %d %d', j, i));
%		end
%	end

%	sg('get_log_likelihood');
%	sg('get_log_likelihood_sample');

