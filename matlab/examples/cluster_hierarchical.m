randn('state',17')
num=400;
k=5;
dist=10;
dims=2;
traindat=[randn(dims,num)-dist, randn(dims,num)+dist, randn(dims,num)+dist*[ones(dims/2,num); zeros(dims/2,num)], randn(dims,num)+dist*[zeros(dims/2,num); ones(dims/2,num)]];
i=randperm(size(traindat,2));
traindat=traindat(:,i);

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('send_command', 'set_distance NORMSQUARED REAL')
sg('send_command', 'init_distance TRAIN');
sg('send_command', 'new_classifier HIERARCHICAL');
tic
sg('send_command', sprintf('train_classifier %d',k));
toc

[assignments,pairs]=sg('get_classifier');

figure()
clf

j=0;
cols='bgrc';
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
