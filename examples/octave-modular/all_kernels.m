init_shogun

% Explicit examples on how to use the different kernels

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');
fm_train_word=uint16(load_matrix('../data/fm_train_word.dat'));
fm_test_word=uint16(load_matrix('../data/fm_test_word.dat'));
fm_train_byte=uint8(load_matrix('../data/fm_train_byte.dat'));
fm_test_byte=uint8(load_matrix('../data/fm_test_byte.dat'));


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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% byte features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% linear byte
disp('LinearByte')

feats_train=ByteFeatures(RAWBYTE);
feats_train.copy_feature_matrix(fm_train_byte);

feats_test=ByteFeatures(RAWBYTE);
feats_test.copy_feature_matrix(fm_test_byte);

kernel=LinearByteKernel(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% chi2
disp('Chi2')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.4;
size_cache=10;

kernel=Chi2Kernel(feats_train, feats_train, width, size_cache);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% const
disp('Const')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
c=23.;

kernel=ConstKernel(feats_train, feats_train, c);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% diag
disp('Diag')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
diag=23.;

kernel=DiagKernel(feats_train, feats_train, diag);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% gaussian
disp('Gaussian')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.9;

kernel=GaussianKernel(feats_train, feats_train, width);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% gaussian_shift
disp('GaussianShift')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.8;
max_shift=2;
shift_step=1;

kernel=GaussianShiftKernel(
	feats_train, feats_train, width, max_shift, shift_step);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% linear
disp('Linear')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
scale=1.2;

kernel=LinearKernel();
kernel.set_normalizer(AvgDiagKernelNormalizer(scale));
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% poly
disp('Poly')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
degree=4;
inhomogene=false;
use_normalization=true;

kernel=PolyKernel(
	feats_train, feats_train, degree, inhomogene, use_normalization);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% sigmoid
disp('Sigmoid')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
size_cache=10;
gamma=1.2;
coef0=1.3;

kernel=SigmoidKernel(feats_train, feats_train, size_cache, gamma, coef0);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% sparse real features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% sparse_gaussian - b0rked?
disp('SparseGaussian')

feat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(feat);
feat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(feat);
width=1.1;

kernel=SparseGaussianKernel(feats_train, feats_train, width);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% sparse_linear
disp('SparseLinear')

feat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(feat);
feat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(feat);
scale=1.1;

kernel=SparseLinearKernel();
kernel.set_normalizer(AvgDiagKernelNormalizer(scale));
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% sparse_poly
disp('SparsePoly')

feat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(feat);
feat=RealFeatures(fm_test_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(feat);
size_cache=10;
degree=3;
inhomogene=true;

kernel=SparsePolyKernel(feats_train, feats_train, size_cache, degree,
	inhomogene);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% word features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% linear_word
disp('LinearWord')

feats_train=WordFeatures(fm_train_word);
feats_test=WordFeatures(fm_test_word);
do_rescale=true;
scale=1.4;

kernel=LinearWordKernel();
kernel.set_normalizer(AvgDiagKernelNormalizer(scale));
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% string features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% fixed_degree_string
disp('FixedDegreeString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
degree=3;

kernel=FixedDegreeStringKernel(feats_train, feats_train, degree);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% linear_string
disp('LinearString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);

kernel=LinearStringKernel(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% local_alignment_strin
disp('LocalAlignmentString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);

kernel=LocalAlignmentStringKernel(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% oligo_string
disp('OligoString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
k=3;
width=1.2;
size_cache=10;

kernel=OligoKernel(size_cache, k, width);
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();


% poly_match_string
disp('PolyMatchString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
degree=3;
inhomogene=false;

kernel=PolyMatchStringKernel(feats_train, feats_train, degree, inhomogene);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% simple_locality_improved_string
disp('SimpleLocalityImprovedString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
l=5;
inner_degree=5;
outer_degree=7;

kernel=SimpleLocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% weighted_degree_string
disp('WeightedDegreeString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
degree=20;

kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

%weights=arange(1,degree+1,dtype=double)[::-1]/ \
%	sum(arange(1,degree+1,dtype=double));
%kernel.set_wd_weights(weights);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% weighted_degree_position_string
disp('WeightedDegreePositionString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
degree=20;

kernel=WeightedDegreePositionStringKernel(feats_train, feats_train, degree);

%kernel.set_shifts(zeros(len(fm_train_dna[0]), dtype=int));

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% locality_improved_string
disp('LocalityImprovedString')

feats_train=StringCharFeatures(DNA);
feats_train.set_string_features(fm_train_dna);
feats_test=StringCharFeatures(DNA);
feats_test.set_string_features(fm_test_dna);
l=5;
inner_degree=5;
outer_degree=7;

kernel=LocalityImprovedStringKernel(
	feats_train, feats_train, l, inner_degree, outer_degree);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% complex string features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

order=3;
gap=0;
reverse=false;

% poly_match_word_string
disp('PolyMatchWordString')

degree=2;
inhomogene=true;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

kernel=PolyMatchWordStringKernel(feats_train, feats_train, degree, inhomogene);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% match_word_string
disp('MatchWordString')

degree=3;
scale=1.4;
size_cache=10;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

kernel=MatchWordStringKernel(size_cache, degree);
kernel.set_normalizer(AvgDiagKernelNormalizer(scale));
kernel.init(feats_train, feats_train);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();


% comm_word_string
disp('CommWordString')

	charfeat=StringCharFeatures(DNA);
	charfeat.set_string_features(fm_train_dna);
	feats_train=StringWordFeatures(charfeat.get_alphabet());
	feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
	preproc=SortWordString();
	preproc.init(feats_train);
	feats_train.add_preproc(preproc);
	feats_train.apply_preproc();

	charfeat=StringCharFeatures(DNA);
	charfeat.set_string_features(fm_test_dna);
	feats_test=StringWordFeatures(charfeat.get_alphabet());
	feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
	feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

kernel=CommWordStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% weighted_comm_word_string
disp('WeightedCommWordString')

order=3;
gap=0;
reverse=true;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

kernel=WeightedCommWordStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% comm_ulong_string
disp('CommUlongString')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringUlongFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortUlongString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();


charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringUlongFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

kernel=CommUlongStringKernel(feats_train, feats_train, use_sign);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% misc kernels
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% custom
%disp('Custom')
%
%dim=7
%data=rand(dim, dim);
%feats=RealFeatures(data);
%symdata=data+data';
%lowertriangle=array([symdata[(x,y)] for x in xrange(symdata.shape[1]);
%	for y in xrange(symdata.shape[0]) if y<=x]);
%
%kernel=CustomKernel(feats, feats);
%
%kernel.set_triangle_kernel_matrix_from_triangle(lowertriangle);
%km_triangletriangle=kernel.get_kernel_matrix();
%
%kernel.set_triangle_kernel_matrix_from_full(symdata);
%km_fulltriangle=kernel.get_kernel_matrix();
%
%kernel.set_full_kernel_matrix_from_full(data);
%km_fullfull=kernel.get_kernel_matrix();

% distance
disp('Distance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.7;
distance=EuclidianDistance();

kernel=DistanceKernel(feats_train, feats_test, width, distance);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% auc
disp('AUC')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.7;
subkernel=GaussianKernel(feats_train, feats_test, width);

num_feats=2; % do not change!
len_train=11;
len_test=17;
data=uint16((len_train-1)*rand(num_feats, len_train));
feats_train=WordFeatures(data);
data=uint16((len_test-1)*rand(num_feats, len_test));
feats_test=WordFeatures(data);

kernel=AUCKernel(feats_train, feats_train, subkernel);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% combined
disp('Combined')

kernel=CombinedKernel();
feats_train=CombinedFeatures();
feats_test=CombinedFeatures();

subkfeats_train=RealFeatures(fm_train_real);
subkfeats_test=RealFeatures(fm_test_real);
subkernel=GaussianKernel(10, 1.2);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

subkfeats_train=StringCharFeatures(DNA);
subkfeats_train.set_string_features(fm_train_dna);
subkfeats_test=StringCharFeatures(DNA);
subkfeats_test.set_string_features(fm_test_dna);
degree=3;
subkernel=FixedDegreeStringKernel(10, degree);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

subkfeats_train=StringCharFeatures(DNA);
subkfeats_train.set_string_features(fm_train_dna);
subkfeats_test=StringCharFeatures(DNA);
subkfeats_test.set_string_features(fm_test_dna);
subkernel=LocalAlignmentStringKernel(10);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);

kernel.init(feats_train, feats_train);
km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

% plugin_estimate
disp('PluginEstimate w/ HistogramWord')

order=3;
gap=0;
reverse=false;

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);

charfeat=StringCharFeatures(DNA);
charfeat.set_string_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);

pie=PluginEstimate();
labels=Labels(label_train_dna);
pie.set_labels(labels);
pie.set_features(feats_train);
pie.train();

kernel=HistogramWordStringKernel(feats_train, feats_train, pie);
km_train=kernel.get_kernel_matrix();

kernel.init(feats_train, feats_test);
pie.set_features(feats_test);
pie.classify().get_labels();
km_test=kernel.get_kernel_matrix();

% top_fisher
disp('TOP/Fisher on PolyKernel')

N=3;
M=6;
pseudo=1e-1;
order=1;
gap=0;
reverse=false;

charfeat=StringCharFeatures(CUBE);
charfeat.set_string_features(cubesequence);
wordfeats_train=StringWordFeatures(charfeat.get_alphabet());
wordfeats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(wordfeats_train);
wordfeats_train.add_preproc(preproc);
wordfeats_train.apply_preproc();

charfeat=StringCharFeatures(CUBE);
charfeat.set_string_features(cubesequence);
wordfeats_test=StringWordFeatures(charfeat.get_alphabet());
wordfeats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
wordfeats_test.add_preproc(preproc);
wordfeats_test.apply_preproc();

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
