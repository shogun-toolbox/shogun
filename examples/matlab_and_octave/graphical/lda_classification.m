num=ceil(500*rand);
dist=0.2
traindat=[rand(2,num)-dist rand(2,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];

sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('new_classifier', 'LDA');
sg('train_classifier');

mi=min(traindat')-0.5;
ma=max(traindat')+0.5;
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
figure(2)
hold on
plot(traindat(1,trainlab==+1), traindat(2,trainlab==+1),'ro');
plot(traindat(1,trainlab==-1), traindat(2,trainlab==-1),'bx');
colorbar
