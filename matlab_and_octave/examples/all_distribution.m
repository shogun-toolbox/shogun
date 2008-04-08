% Explicit examples on how to use distributions

leng=50;
rep=5;
weight=1;
order=3;
gap=0;
num=12;
len=23;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

% Histogram
disp('Histogram');

%sg('send_command', 'new_distribution HISTOGRAM');
sg('send_command', 'add_preproc SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');

%	sg('send_command', 'train_distribution');
%	histo=sg('get_histogram');

%	num_param=sg('get_histogram_num_model_parameters');
%	for i = 1:num,
%		for j = 1:num_param,
%			sg(sprintf('get_log_derivative %d %d', j, i));
%		end
%	end

%	sg('get_log_likelihood');
%	sg('get_log_likelihood_sample');

% LinearHMM
disp('LinearHMM');

%sg('send_command', 'new_distribution LinearHMM');
sg('send_command', 'add_preproc SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');

%	sg('send_command', 'train_distribution');
%	histo=sg('get_histogram');

%	num_param=sg('get_histogram_num_model_parameters');
%	for i = 1:num,
%		for j = 1:num_param,
%			sg(sprintf('get_log_derivative %d %d', j, i));
%		end
%	end

%	sg('get_log_likelihood');
%	sg('get_log_likelihood_sample');

% HMM
disp('HMM');

N=3;
M=6;

% generate a sequence with characters 1-6 drawn from 3 loaded cubes
for i = 1:3,
    a{i}= [ ones(1,ceil(leng*rand)) 2*ones(1,ceil(leng*rand)) 3*ones(1,ceil(leng*rand)) 4*ones(1,ceil(leng*rand)) 5*ones(1,ceil(leng*rand)) 6*ones(1,ceil(leng*rand)) ];
    a{i}= a{i}(randperm(length(a{i})));
end

s=[];
for i = 1:size(a,2),
    s= [ s i*ones(1,ceil(rep*rand)) ];
end
s=s(randperm(length(s)));
sequence={''};
for i = 1:length(s),
    f(i)=ceil(((1-weight)*rand+weight)*length(a{s(i)}));
    t=randperm(length(a{s(i)}));
    r=a{s(i)}(t(1:f(i)));
    sequence{1}=[sequence{1} char(r+'0')];
end


sg('send_command', sprintf('new_hmm %d %d', N, M));
sg('set_features','TRAIN', sequence, 'CUBE');
sg('send_command', 'convert TRAIN STRING CHAR STRING WORD 1');
sg('send_command', 'bw');
[p, q, a, b]=sg('get_hmm');

sg('send_command', sprintf('new_hmm %d %d', N, M));
sg('set_hmm', p, q, a, b);
sg('set_features','TRAIN', sequence,'CUBE');
likelihood=sg('hmm_likelihood')


