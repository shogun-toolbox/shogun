num=1000;
dim=100;

%sg('loglevel', 'ALL');
rand('state',sum(100*clock));
traindat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
testlab=[-ones(1,num/2), ones(1,num/2) ];

%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('set_kernel', 'GAUSSIAN', 'REAL', 100' 1000);
%sg('new_classifier', 'GPBTSVM');
%sg('c', 2);
%sg('train_classifier');
%%[b, alphas]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%out=sg('classify');
%valerr=mean(testlab~=sign(out));
%
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('set_kernel', 'GAUSSIAN', 'REA', 100, 1000);
%sg('new_classifier', 'LIBSVM');
%sg('c', 2);
%sg('train_classifier');
%%[b2, alphas2]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%out2=sg('classify');
%valerr2=mean(testlab~=sign(out2));
%
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('new_classifier', 'PERCEPTRON');
%sg('set_perceptron_parameters', 0.001, 10000);
%sg('train_classifier');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%out3=sg('classify');
%valerr3=mean(testlab~=sign(out3))
%sg('set_features', 'TEST', traindat);
%sg('set_labels', 'TEST', trainlab);
%outt3=sg('classify');
%valerrt3=mean(trainlab~=sign(outt3))

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('new_classifier', 'LDA');
sg('train_classifier');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out4=sg('classify');
valerr4=mean(testlab~=sign(out4))
sg('set_features', 'TEST', traindat);
sg('set_labels', 'TEST', trainlab);
outt4=sg('classify');
valerrt4=mean(trainlab~=sign(outt4))
