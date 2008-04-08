traindat = (1:100)/10;
trainlab = sin(traindat);
testdat = ((1:100)-0.5)/10;
testlab = sin(testdat);

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel GAUSSIAN REAL 10 0.1');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIBSVR');
sg('send_command', 'c 0.1');
sg('send_command', 'svr_tube_epsilon 0.2');
sg('send_command', 'svm_train');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
sg('send_command', 'init_kernel TEST');
out=sg('svm_classify');
err=mean((testlab-out).^2);

figure(1)
clf
plot(traindat, trainlab,'bo-')
hold on
plot(testdat, testlab,'ro-')
