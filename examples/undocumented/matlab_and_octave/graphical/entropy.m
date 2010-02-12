order = 1; %markov chain (1 == zeroth order)
ppseudo=1e-3; % pseudo counts
npseudo=1e-3; % pseudo counts

acgt='ACGT';

XT=char(acgt(ceil(4*rand(506,10000))));
LT=[ones(1,5000) -ones(1,5000)];
XV=char(acgt(ceil(4*rand(506,10000))));
LV=[ones(1,5000) -ones(1,5000)];

XT(250:285,1:5000)='A';
XV(250:285,1:5000)='A';

% XT -> train data
% LT -> train label
% XV, LV - validdation

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', XT(:,LT==1), 'DNA') ;
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);
sg('pseudo', ppseudo);
sg('new_hmm', size(XT,1)-order+1, 4^order);
sg('linear_train');
[p_p,q_p,a_p,b_p]=sg('get_hmm');
sg('set_features', 'TEST', XV , 'DNA') ;
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1);

posout=sg('one_class_linear_hmm_classify');
ee=sg('entropy');
sg('set_hmm_as', 'POS');

sg('set_features', 'TRAIN', XT(:,LT==-1), 'DNA') ;

sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD',order, order-1);
sg('pseudo', npseudo);
sg('new_hmm', size(XT,1)-order+1, 4^order);
sg('linear_train');
[p_n,q_n,a_n,b_n]=sg('get_hmm');
sg('set_features', 'TEST', XV, 'DNA') ;
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD',order, order-1);

negout=sg('one_class_linear_hmm_classify');
sg('set_hmm_as', 'NEG');
relee=sg('relative_entropy');

output=posout-negout;

figure(1);
plot(relee)
title('relative entropy')

figure(2);
plot(ee);
title('entropy')
