num=500;
dist=2
traindat=[rand(2,num)-dist rand(2,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'LINEAR', 'REAL', 1, 1.0);
%sg('set_kernel', 'GAUSSIAN', 'REAL', 100, 500);
sg('new_classifier', 'GNPPSVM');
sg('svm_epsilon', 1e-10);
sg('c', 2000000);
tic;
sg('train_classifier');
toc

[npp_b npp_alpha]=sg('get_svm');

mi=min(traindat,2)-0.2;
ma=max(traindat,2)+0.2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
npp_out=sg('classify');
[npp_b npp_alpha]=sg('get_svm');

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'LINEAR', 'REAL', 1, 1.0);
%sg(set_kernel', 'GAUSSIAN', 'REAL', 100, 500);
sg('new_classifier', 'LIBSVM');
sg('svm_epsilon', 1e-10);
sg('c', 2000000);
tic;
sg('train_classifier');
toc

mi=min(traindat,2)-0.2;
ma=max(traindat,2)+0.2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
lib_out=sg('classify');
[lib_b lib_alpha]=sg('get_svm');

figure(1)
clf
out=reshape(npp_out,50,50);
pcolor(out)
shading interp
hold on
contour(out,'k-')
colorbar

figure(2)
clf
out=reshape(lib_out,50,50);
pcolor(out)
shading interp
hold on
contour(out,'k-')
colorbar

figure(3)
clf
plot(traindat(1,:),traindat(2,:),'.')

max(abs(lib_out-npp_out))
lib_out(1:10)-npp_out(1:10)

