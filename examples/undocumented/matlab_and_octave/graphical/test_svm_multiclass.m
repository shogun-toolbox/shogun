%4 class toy problem with L2 b-svm
%rand('state',17);
num=100;
dist=10
traindat=[rand(2,num)-dist rand(2,num) rand(2,num)+dist rand(2,num)+2*dist];
trainlab=[zeros(1,num) ones(1,num) 2*ones(1,num) 3*ones(1,num)];

idx=randperm(length(trainlab));
traindat=traindat(:,idx);
trainlab=trainlab(idx);

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100, 2);
sg('new_classifier', 'LIBSVM_MULTICLASS');
%sg('new_classifier', 'GMNP');
sg('svm_epsilon', 1e-5)
sg('c', 10);
sg('train_classifier');

mi=min(traindat')-2;
ma=max(traindat')+2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
out=sg('classify');

figure(1)
clf
out=reshape(out,50,50);
pcolor(out)
shading interp
hold on
contour(out,'k-')
colorbar

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100, 2);
%sg('new_classifier', 'LIBSVM_MULTICLASS');
sg('new_classifier', 'GMNPSVM');
sg('svm_epsilon', 1e-5);
sg('c', 10);
sg('train_classifier');

mi=min(traindat')-2;
ma=max(traindat')+2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
out2=sg('classify');

figure(2)
clf
out2=reshape(out2,50,50);
pcolor(out2)
shading interp
hold on
contour(out2,'k-')
colorbar
