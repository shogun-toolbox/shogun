init_shogun

addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');

leng=50;
rep=5;
weight=0.3;

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
cubesequence={''};
for i = 1:length(s),
    f(i)=ceil(((1-weight)*rand+weight)*length(a{s(i)}));
    t=randperm(length(a{s(i)}));
    r=a{s(i)}(t(1:f(i)));
    cubesequence{1}=[cubesequence{1} char(r+'0')];
end

% Histogram
disp('Histogram')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_train_dna);
feats=StringWordFeatures(charfeat.get_alphabet());
feats.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats);
feats.add_preproc(preproc);
feats.apply_preproc();

histo=Histogram(feats);
histo.train();

histo.get_histogram();

num_examples=feats.get_num_vectors();
num_param=histo.get_num_model_parameters();
for i=0:(num_examples-1),
	for j=0:(num_param-1),
		histo.get_log_derivative(j, i);
	end
end

histo.get_log_likelihood();
histo.get_log_likelihood_sample();
