randn('state',17')
num=400;
k=5;
dist=10;
dims=2;
traindat=[randn(dims,num)-dist, randn(dims,num)+dist, randn(dims,num)+dist*[ones(dims/2,num); zeros(dims/2,num)], randn(dims,num)+dist*[zeros(dims/2,num); ones(dims/2,num)]];
i=randperm(size(traindat,2));
traindat=traindat(:,i);

sg('loglevel', 'ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_distance', 'EUCLIDEAN', 'REAL')
sg('new_clustering', 'HIERARCHICAL');
tic
sg('train_clustering', k);
toc

[assignments,pairs]=sg('get_clustering');

figure()
clf

j=0;
cols='bgrcm';  % each of k=5 clusters has its own color
ii=unique(assignments(:));
for i=1:length(ii),
	j=j+1;
	idx=find(assignments(:)==ii(i));
	plot(traindat(1,idx), traindat(2,idx),sprintf('%cx',cols(j)));
	hold on
end

figure
imagesc(traindat')

figure
[dummy,idx]=sort(assignments);
sdat=traindat(:,idx);
imagesc(sdat');
