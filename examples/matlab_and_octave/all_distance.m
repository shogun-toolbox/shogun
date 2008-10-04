% Explicit examples on how to use the different distances;

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');


%
% real features
%

% BrayCurtis Distance
disp('BrayCurtisDistance');
sg('set_distance', 'BRAYCURTIS', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);;
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');

% Euclidian Distance
disp('EuclidianDistance');
sg('set_distance', 'EUCLIDIAN', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);;
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Canberra Metric
disp('CanberraMetric');
sg('set_distance', 'CANBERRA', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Chebyshew Metric
disp('ChebyshewMetric');
sg('set_distance', 'CHEBYSHEW', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Chi Square Metric
disp('ChiSquareDistance');
sg('set_distance', 'CHISQUARE', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Cosine Distance
disp('CosineDistance');
sg('set_distance', 'COSINE', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Geodesic Metric
disp('GeodesicMetric');
sg('set_distance', 'GEODESIC', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Jensen Metric
disp('JensenMetric');
sg('set_distance', 'JENSEN', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Manhattan Metric
disp('ManhattanMetric');
sg('set_distance', 'MANHATTAN', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Minkowski Metric
disp('MinkowskiMetric');
k=3;
sg('set_distance', 'MINKOWSKI', 'REAL', k);

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Tanimoto Metric
disp('TanimotoDistance');
sg('set_distance', 'TANIMOTO', 'REAL');

sg('set_features', 'TRAIN', fm_train_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');



%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true

% CanberraWord Distance
disp('CanberraWordDistance');

sg('set_distance', 'CANBERRA', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% HammingWord Distance
disp('HammingWordDistance');

sg('set_distance', 'HAMMING', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% ManhattanWord Distance
disp('ManhattanWordDistance');

sg('set_distance', 'MANHATTAN', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');

