num=1000;
dim=100;

sg('send_command', 'loglevel ALL');
rand('state',sum(100*clock));
traindat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
trainlab=[-ones(1,num/2), ones(1,num/2) ];

testdat=[rand(dim,num/2)-0.05, rand(dim,num/2)+0.05];
testlab=[-ones(1,num/2), ones(1,num/2) ];

%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', 'set_kernel GAUSSIAN REAL 100 1000');
%sg('send_command', 'init_kernel TRAIN');
%sg('send_command', 'new_classifier GPBTSVM');
%sg('send_command', 'c 2');
%sg('send_command', 'train_classifier');
%%[b, alphas]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%sg('send_command', 'init_kernel TEST');
%out=sg('classify');
%valerr=mean(testlab~=sign(out));
%
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', 'set_kernel GAUSSIAN REAL 100 1000');
%sg('send_command', 'init_kernel TRAIN');
%sg('send_command', 'new_classifier LIBSVM');
%sg('send_command', 'c 2');
%sg('send_command', 'train_classifier');
%%[b2, alphas2]=sg('get_svm');
%sg('set_features', 'TEST', testdat);
%sg('set_labels', 'TEST', testlab);
%sg('send_command', 'init_kernel TEST');
%out2=sg('classify');
%valerr2=mean(testlab~=sign(out2));
%
%sg('set_features', 'TRAIN', traindat);
%sg('set_labels', 'TRAIN', trainlab);
%sg('send_command', 'new_classifier PERCEPTRON');
%sg('send_command', 'set_perceptron_parameters 0.001 10000');
%sg('send_command', 'train_classifier');
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
sg('send_command', 'new_classifier LDA');
sg('send_command', 'train_classifier');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out4=sg('classify');
valerr4=mean(testlab~=sign(out4))
sg('set_features', 'TEST', traindat);
sg('set_labels', 'TEST', trainlab);
outt4=sg('classify');
valerrt4=mean(trainlab~=sign(outt4))
