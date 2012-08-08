k=4;
num=1000;
iter=50000;
dist=2.2;
traindat=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
trainlab=[ones(1,num) 2*ones(1,num) 3*ones(1,num) 4*ones(1,num)];

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_distance', 'EUCLIDEAN', 'REAL')
sg('new_clustering', 'KMEANS');
sg('train_clustering', k, iter);

[radi,centers]=sg('get_clustering');

figure()
clf
plot(traindat(1,trainlab==+1), traindat(2,trainlab==+1),'rx');
hold on
plot(traindat(1,trainlab==+2), traindat(2,trainlab==+2),'bx');
plot(traindat(1,trainlab==+3), traindat(2,trainlab==+3),'gx');
plot(traindat(1,trainlab==+4), traindat(2,trainlab==+4),'cx');

plot(centers(1,:), centers(2,:), 'ko');

for i=1:k
	t = linspace(0, 2*pi, 100);
	plot(radi(i)*cos(t)+centers(1,i),radi(i)*sin(t)+centers(2,i),'k-')
end

