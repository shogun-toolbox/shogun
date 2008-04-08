% Explicit examples on how to use the different preprocs

num=20;
len=42;
size_cache=10;
width=1.4;
dist=3.3;

%
% real features
%

traindata_real=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
testdata_real=[randn(2,num+7)-dist, randn(2,num+7)+dist, randn(2,num+7)+dist*[ones(1,num+7); zeros(1,num+7)], randn(2,num+7)+dist*[zeros(1,num+7); ones(1,num+7)]];


% LogPlusOne
disp('LogPlusOne');

sg('send_command', 'add_preproc LOGPLUSONE');
sg('send_command', sprintf('set_kernel CHI2 REAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');

% NormOne
disp('NormOne');

sg('send_command', 'add_preproc NORMONE');
sg('send_command', sprintf('set_kernel CHI2 REAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% PruneVarSubMean
disp('PruneVarSubMean');

divide_by_std=1;

sg('send_command', sprintf('add_preproc PRUNEVARSUBMEAN %d', divide_by_std));
sg('send_command', sprintf('set_kernel CHI2 REAL %d %f', size_cache, width));

sg('set_features', 'TRAIN', traindata_real);
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_real);
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');

%
% word features;
%

maxval=2^16-1;
traindata_word=uint16(rand(len, num)*maxval);
testdata_word=uint16(rand(len, num)*maxval);

% LinearWord
disp('LinearWord');

scale=1.4;

sg('send_command', 'add_preproc SORTWORD');
sg('send_command', sprintf('set_kernel LINEAR WORD %d %f', size_cache, scale));

sg('set_features', 'TRAIN', traindata_word);
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_word);
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


%
% complex string features;
%

order=3;
gap=0;
reverse='n'; % bit silly to not use boolean, set 'r' to yield true
use_sign=0;
normalization='FULL';

acgt='ACGT';
trainlab_dna=[ones(1,num/2) -ones(1,num/2)];
traindata_dna=acgt(ceil(4*rand(len,num)));
testdata_dna=acgt(ceil(4*rand(len,num)));

% SortWordString
disp('CommWordString');

sg('send_command', 'add_preproc SORTWORDSTRING');
sg('send_command', sprintf('set_kernel COMMSTRING WORD %d %d %s', size_cache, use_sign, normalization));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING WORD %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING WORD %d %d %d %c' , order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');
sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


% SortUlongString
disp('CommUlongString');

sg('send_command', 'add_preproc SORTULONGSTRING');
sg('send_command', sprintf('set_kernel COMMSTRING ULONG %d %d %s', size_cache, use_sign, normalization));

sg('set_features', 'TRAIN', traindata_dna, 'DNA');
sg('send_command', sprintf('convert TRAIN STRING CHAR STRING ULONG %d %d %d %c', order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TRAIN');
sg('send_command', 'init_kernel TRAIN');
km=sg('get_kernel_matrix');

sg('set_features', 'TEST', testdata_dna, 'DNA');
sg('send_command', sprintf('convert TEST STRING CHAR STRING ULONG %d %d %d %c' , order, order-1, gap, reverse));
sg('send_command', 'attach_preproc TEST');

sg('send_command', 'init_kernel TEST');
km=sg('get_kernel_matrix');


