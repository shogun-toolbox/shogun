randn('state',17')
num=200;
k=4;
dist=8;
traindat=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
%trainlab=[ones(1,num) 2*ones(1,num) 3*ones(1,num) 4*ones(1,num)];

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

dist=sg('get_distance_matrix');
figure
imagesc(dist)

%figure
%l=length(pairs,2);
%sdist=zeros(l);
%for i=1:l,
%end
