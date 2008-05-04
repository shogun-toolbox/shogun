init_shogun

num=50; %number of example
len=10; %number of dimensions
dist=1.5;

% Explicit examples on how to use the different preprocs

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

traindata_real=[randn(len,num)-dist, randn(len,num)+dist];
testdata_real=[randn(len,num+7)-dist, randn(len,num+7)+dist];

maxval=2^16-1;
traindata_word=uint16(rand(len, num)*maxval);
testdata_word=uint16(rand(len, num)*maxval);


%LogPlusOne
disp('LogPlusOne')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

preproc=LogPlusOne();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%NormOne
disp('NormOne')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

preproc=NormOne();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%PruneVarSubMean
disp('PruneVarSubMean')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

preproc=PruneVarSubMean();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% word features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%LinearWord
disp('LinearWord')

feats_train=WordFeatures(traindata_word);
feats_test=WordFeatures(testdata_word);

preproc=SortWord();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

do_rescale=true;
scale=1.4;

kernel=LinearWordKernel(feats_train, feats_train, do_rescale, scale);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% complex string features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%CommWordString
disp('CommWordString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(traindata_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(testdata_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;
normalization=FULL_NORMALIZATION;

kernel=CommWordStringKernel(
	feats_train, feats_train, use_sign, normalization);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%CommUlongString
disp('CommUlongString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(traindata_dna);
feats_train=StringUlongFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(testdata_dna);
feats_test=StringUlongFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

preproc=SortUlongString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;
normalization=FULL_NORMALIZATION;

kernel=CommUlongStringKernel(
	feats_train, feats_train, use_sign, normalization);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();
