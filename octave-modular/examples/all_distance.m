init_shogun

len=17;
num=42;
dist=2.3;

% Explicit examples on how to use the different distances
traindata_real=[randn(2,num)-dist, randn(2,num)+dist];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist];

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% euclidian distance
disp('EuclidianDistance')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=EuclidianDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% norm squared distance
disp('EuclidianDistance - NormSquared')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=EuclidianDistance(feats_train, feats_train);
distance.set_disable_sqrt(true);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% canberra metric
disp('CanberaMetric')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=CanberraMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% chebyshew metric
disp('ChebyshewMetric')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=ChebyshewMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% geodesic metric
disp('GeodesicMetric')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=GeodesicMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% jensen metric
disp('JensenMetric')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=JensenMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% manhattan metric
disp('ManhattanMetric')

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=ManhattanMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% minkowski metric
disp('MinkowskiMetric')

k=3

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);

distance=MinkowskiMetric(feats_train, feats_train, k);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% sparse euclidian distance
disp('SparseEuclidianDistance')

realfeat=RealFeatures(traindata_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(testdata_real);
feats_test=SparseRealFeatures();
feats_test.obtain_from_simple(realfeat);

distance=SparseEuclidianDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% complex string features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% canberra word distance
disp('CanberraWordDistance')

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

distance=CanberraWordDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% hamming word distance
disp('HammingWordDistance')

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

use_sign=false

distance=HammingWordDistance(feats_train, feats_train, use_sign);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% manhattan word distance
disp('ManhattanWordDistance')

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

distance=ManhattanWordDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();
