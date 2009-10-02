addpath('tools');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

order=3;
gap=0;
reverse='n';

% CanberraWord Distance
disp('CanberraWordDistance');

sg('set_distance', 'CANBERRA', 'WORD');
sg('add_preproc', 'SORTWORDSTRING');

sg('set_features', 'TRAIN', fm_train_dna, 'DNA');
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TRAIN');
dm=sg('get_distance_matrix', 'TRAIN');

sg('set_features', 'TEST', fm_test_dna, 'DNA');
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1, gap, reverse);
sg('attach_preproc', 'TEST');
dm=sg('get_distance_matrix', 'TEST');
