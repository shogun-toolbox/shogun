init_shogun


% Explicit examples on how to use the different distances

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% real features
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% bray curtis distance
disp('BrayCurtisDistance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=BrayCurtisDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% euclidian distance
disp('EuclidianDistance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=EuclidianDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% norm squared distance
disp('EuclidianDistance - NormSquared')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=EuclidianDistance(feats_train, feats_train);
distance.set_disable_sqrt(true);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% canberra metric
disp('CanberaMetric')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=CanberraMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% chebyshew metric
disp('ChebyshewMetric')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=ChebyshewMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% chi square distance
disp('ChiSquareDistance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=ChiSquareDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% cosine distance
disp('Cosine Distance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=CosineDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();


% geodesic metric
disp('GeodesicMetric')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=GeodesicMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% jensen metric
disp('JensenMetric')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=JensenMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% manhattan metric
disp('ManhattanMetric')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=ManhattanMetric(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% minkowski metric
disp('MinkowskiMetric')

k=3;

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=MinkowskiMetric(feats_train, feats_train, k);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();

% tanimoto distance
disp('TanimotoDistance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);

distance=TanimotoDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();


% sparse euclidian distance
disp('SparseEuclidianDistance')

realfeat=RealFeatures(fm_train_real);
feats_train=SparseRealFeatures();
feats_train.obtain_from_simple(realfeat);
realfeat=RealFeatures(fm_test_real);
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
charfeat.set_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
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
charfeat.set_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

use_sign=false;

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
charfeat.set_features(fm_train_dna);
feats_train=StringWordFeatures(charfeat.get_alphabet());
feats_train.obtain_from_char(charfeat, order-1, order, gap, reverse);
preproc=SortWordString();
preproc.init(feats_train);
feats_train.add_preproc(preproc);
feats_train.apply_preproc();

charfeat=StringCharFeatures(DNA);
charfeat.set_features(fm_test_dna);
feats_test=StringWordFeatures(charfeat.get_alphabet());
feats_test.obtain_from_char(charfeat, order-1, order, gap, reverse);
feats_test.add_preproc(preproc);
feats_test.apply_preproc();

distance=ManhattanWordDistance(feats_train, feats_train);

dm_train=distance.get_distance_matrix();
distance.init(feats_train, feats_test);
dm_test=distance.get_distance_matrix();
