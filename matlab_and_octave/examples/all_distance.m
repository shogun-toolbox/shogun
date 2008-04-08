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
sg('send_command', 'set_distance EUCLIDIAN REAL');

sg('set_features', 'TRAIN', traindata_real);;
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);;
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Canberra Metric
disp('CanberaMetric');
sg('send_command', 'set_distance CANBERRA REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Chebyshew Metric
disp('ChebyshewMetric');
sg('send_command', 'set_distance CHEBYSHEW REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Geodesic Metric
disp('GeodesicMetric');
sg('send_command', 'set_distance GEODESIC REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Jensen Metric
disp('JensenMetric');
sg('send_command', 'set_distance JENSEN REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Manhattan Metric
disp('ManhattanMetric');
sg('send_command', 'set_distance MANHATTAN REAL');

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% Minkowski Metric
disp('MinkowskiMetric');
k=3;
sg('send_command', sprintf('set_distance MINKOWSKI REAL %d', k));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'init_distance TEST');
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

sg('send_command', 'set_distance CANBERRA WORD');
sg('send_command', 'add_preproc SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% HammingWord Distance
disp('HammingWordDistance');

sg('send_command', 'set_distance HAMMING WORD');
sg('send_command', 'add_preproc SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');


% ManhattanWord Distance
disp('ManhattanWordDistance');

sg('send_command', 'set_distance MANHATTAN WORD');
sg('send_command', 'add_preproc SORTWORDSTRING');

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_distance TRAIN');
dm=sg('get_distance_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_distance TEST');
dm=sg('get_distance_matrix');

