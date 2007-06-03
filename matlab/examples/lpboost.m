C=100;
epsilon=1e-3;

%load ~/subgradient/data/astro-ph_29882.mat
%load ~/subgradient/data/astro-ph_full.mat
%traindat=x;
%trainlab=y;
%clear x;
%clear y;

load ~/subgradient/data/uci_spambase.mat
traindat=sparse(x');
trainlab=t';

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf('c %f', C));
sg('send_command', 'svm_use_bias 0');
sg('send_command', 'new_classifier LPBOOST');
tic;
sg('send_command', 'train_classifier');
timelpm=toc

[b,W]=sg('get_classifier');

sg('set_features', 'TEST', traindat);
trainout=sg('classify');
trainerr=mean(trainlab~=sign(trainout))

b
%W'
