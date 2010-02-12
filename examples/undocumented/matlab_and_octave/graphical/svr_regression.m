traindat = (1:100)/10;
trainlab = sin(traindat);
testdat = ((1:100)-0.5)/10;
testlab = sin(testdat);

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 10, 0.1);
sg('new_regression', 'LIBSVR');
sg('c', 0.1);
sg('svr_tube_epsilon', 0.2);
sg('train_regression');
sg('set_features', 'TEST', testdat);
sg('set_labels', 'TEST', testlab);
out=sg('classify');
err=mean((testlab-out).^2);

figure(1)
clf
plot(traindat, trainlab,'bo-')
hold on
plot(testdat, testlab,'ro-')
