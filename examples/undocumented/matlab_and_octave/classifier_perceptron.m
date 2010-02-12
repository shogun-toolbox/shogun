addpath('tools');

% Perceptron
disp('Perceptron');

% create some seperable toy data
num=50;
label_train_twoclass=[-ones(1,num/2) ones(1,num/2)];
fm_train_real=[randn(5,num/2)-1, randn(5,num/2)+1];
fm_test_real=[randn(5,num)-1, randn(5,num)+1];

sg('set_features', 'TRAIN', fm_train_real);
sg('set_labels', 'TRAIN', label_train_twoclass);
sg('new_classifier', 'PERCEPTRON');
%sg('set_perceptron_parameters', 1.6, 5000);
sg('train_classifier');

sg('set_features', 'TEST', fm_test_real);
result=sg('classify');
