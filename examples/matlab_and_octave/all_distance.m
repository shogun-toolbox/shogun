% Explicit examples on how to use the different distances;

num=24;
len=42;

%
% real features
%

dist=2.4;
traindata_real=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];

% Euclidian Distance
disp('EuclidianDistance');
sg('set_distance', 'EUCLIDIAN', 'REAL');

sg('set_features', 'TRAIN', traindata_real);;
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Canberra Metric
disp('CanberraMetric');
sg('set_distance', 'CANBERRA', 'REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Chebyshew Metric
disp('ChebyshewMetric');
sg('set_distance', 'CHEBYSHEW', 'REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Geodesic Metric
disp('GeodesicMetric');
sg('set_distance', 'GEODESIC', 'REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Jensen Metric
disp('JensenMetric');
sg('set_distance', 'JENSEN', 'REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Manhattan Metric
disp('ManhattanMetric');
sg('set_distance', 'MANHATTAN', 'REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% Minkowski Metric
disp('MinkowskiMetric');
k=3;
sg('set_distance', 'MINKOWSKI', 'REAL', k);

sg('set_features', 'TRAIN', traindata_real);
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

% CanberraWord Distance
disp('CanberraWordDistance');

sg('set_distance', 'CANBERRA', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% HammingWord Distance
disp('HammingWordDistance');

sg('set_distance', 'HAMMING', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');


% ManhattanWord Distance
disp('ManhattanWordDistance');

sg('set_distance', 'MANHATTAN', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
sg('init_distance', 'TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
sg('init_distance', 'TEST');
dm=sg('get_distance_matrix');

