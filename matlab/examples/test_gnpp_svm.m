num=500;
dist=2
traindat=[rand(2,num)-dist rand(2,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel LINEAR REAL 1 1.0');
%sg('send_command', 'set_kernel GAUSSIAN REAL 100 500');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm GNPP');
sg('send_command','svm_epsilon 1e-10');
sg('send_command', 'c 2000000');
tic;
sg('send_command', 'svm_train');
toc

[npp_b npp_alpha]=sg('get_svm');

mi=min(traindat,2)-0.2;
ma=max(traindat,2)+0.2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
sg('send_command', 'init_kernel TEST');
npp_out=sg('svm_classify');
[npp_b npp_alpha]=sg('get_svm');

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_kernel LINEAR REAL 1 1.0');
%sg('send_command', 'set_kernel GAUSSIAN REAL 100 500');
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIBSVM');
sg('send_command','svm_epsilon 1e-10')
sg('send_command', 'c 2000000');
tic;
sg('send_command', 'svm_train');
toc

mi=min(traindat,2)-0.2;
ma=max(traindat,2)+0.2;
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
sg('send_command', 'init_kernel TEST');
lib_out=sg('svm_classify');
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

