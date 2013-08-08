modshogun

addpath('tools');

leng=28;
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

% top_fisher
disp('TOP/Fisher on PolyKernel')

N=3;
M=6;
pseudo=1e-1;
order=1;
gap=0;
reverse=false;

charfeat=StringCharFeatures(CUBE);
charfeat.set_features(cubesequence);
wordfeats_train=StringWordFeatures(charfeat.get_alphabet());
wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(wordfeats_train);
wordfeats_train.add_preprocessor(preproc);
wordfeats_train.apply_preprocessor();

charfeat=StringCharFeatures(CUBE);
charfeat.set_features(cubesequence);
wordfeats_test=StringWordFeatures(charfeat.get_alphabet());
wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
wordfeats_test.add_preprocessor(preproc);
wordfeats_test.apply_preprocessor();

% cheating, BW_NORMAL is somehow not available
BW_NORMAL=0;
pos=HMM(wordfeats_train, N, M, pseudo);
pos.train();
pos.baum_welch_viterbi_train(BW_NORMAL);
neg=HMM(wordfeats_train, N, M, pseudo);
neg.train();
neg.baum_welch_viterbi_train(BW_NORMAL);
pos_clone=HMM(pos);
neg_clone=HMM(neg);
pos_clone.set_observations(wordfeats_test);
neg_clone.set_observations(wordfeats_test);

feats_train=TOPFeatures(10, pos, neg, false, false);
feats_test=TOPFeatures(10, pos_clone, neg_clone, false, false);
kernel=PolyKernel(feats_train, feats_train, 1, false, true);
km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

feats_train=FKFeatures(10, pos, neg);
feats_train.set_opt_a(-1); %estimate prior
feats_test=FKFeatures(10, pos_clone, neg_clone);
feats_test.set_a(feats_train.get_a()); %use prior from training data
kernel=PolyKernel(feats_train, feats_train, 1, false, true);
km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
