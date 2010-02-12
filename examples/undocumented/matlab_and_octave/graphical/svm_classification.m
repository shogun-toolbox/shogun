num=100;
dist=0.8;
traindat=[randn(2,num)-dist randn(2,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('set_kernel', 'GAUSSIAN', 'REAL', 100, 1.0);
sg('new_classifier', 'LIBSVM');
sg('svm_epsilon', 1e-3)
sg('c', 20);
sg('train_classifier');

[b,alphas]=sg('get_svm');
svidx=alphas(:,2)+1;

mi=[min(traindat(1,:))-0.2, min(traindat(2,:))-0.2];
ma=[max(traindat(1,:))+0.2, max(traindat(2,:))+0.2];
[x,y]=meshgrid(linspace(mi(1),ma(1),50), linspace(mi(2),ma(2),50));
testdat=[x(:),y(:)]';
sg('set_features', 'TEST', testdat);
out=sg('classify');

figure(1)
clf
out=reshape(out,50,50);
pcolor(x,y,out)
shading interp
hold on
contour(x,y,out,'k-')
colorbar

pidx=find(trainlab>0);
pidx=setdiff(pidx, svidx);
plot(traindat(1,pidx), traindat(2,pidx), 'k*','MarkerSize',3);

nidx=find(trainlab<0);
nidx=setdiff(nidx, svidx);
plot(traindat(1,nidx), traindat(2,nidx), 'k.','MarkerSize',3);

pidx=find(trainlab>0);
pidx=intersect(pidx, svidx);
plot(traindat(1,pidx), traindat(2,pidx), 'r*','MarkerSize',3);

nidx=find(trainlab<0);
nidx=intersect(nidx, svidx);
plot(traindat(1,nidx), traindat(2,nidx), 'b.','MarkerSize',3);
