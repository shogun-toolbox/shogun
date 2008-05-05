%generate some toy data
acgt='ACGT';
dat={acgt([1*ones(1,10) 2*ones(1,10) 3*ones(1,10) 4*ones(1,10)])};
sg('set_features', 'TRAIN', dat, 'DNA');
sg('slide_window', 'TRAIN', 5, 1');
