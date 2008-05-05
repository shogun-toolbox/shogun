%generate some toy data
acgt='ACGT';
dat={acgt([1*ones(1,10) 2*ones(1,10) 3*ones(1,10) 4*ones(1,10) 1])};
sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', dat, 'DNA');
sg('slide_window', 'TRAIN', 5, 1);

f=sg('get_features', 'TRAIN')


sg('set_features', 'TRAIN', dat, 'DNA');
sg('from_position_list','TRAIN', 5, int32([0,1,2,5,15,25,30,36]));

f=sg('get_features', 'TRAIN')

sg('set_features', 'TEST', dat, 'DNA');
sg('from_position_list','TEST', 5, int32([0,1,2,5,15,25,30,36]));

ft=sg('get_features', 'TEST')

C=1;
order=20;
order_com=5;
max_mismatch=0;
len=200;
shift=0;
num=100;
num_test=5000;
cache=10;
normalize=1;
mkl_stepsize=1;
block=0;
single_degree=-1;
sg('set_kernel', 'WEIGHTEDDEGREE', 'STRING', cache, order, max_mismatch, normalize, mkl_stepsize, block, single_degree);
sg('init_kernel', 'TRAIN');
km=sg('get_kernel_matrix')

sg('clean_features', 'TRAIN');
sg('clean_features', 'TEST');
sg('set_features', 'TRAIN', dat, 'DNA');
sg('from_position_list','TRAIN', 5, int32([0,1,2,5,15,25,30]+5));
sg('set_features', 'TRAIN', dat, 'DNA');
sg('from_position_list','TRAIN', 5, int32([0,1,2,5,15,25]+9));
sg('clean_features', 'TRAIN');
