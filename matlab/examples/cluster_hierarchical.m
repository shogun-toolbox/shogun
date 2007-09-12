merges=40;
num=10;
dist=100;
traindat=[randn(2,num)-dist, randn(2,num)+dist, randn(2,num)+dist*[ones(1,num); zeros(1,num)], randn(2,num)+dist*[zeros(1,num); ones(1,num)]];
%trainlab=[ones(1,num) 2*ones(1,num) 3*ones(1,num) 4*ones(1,num)];

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('send_command', 'set_distance NORMSQUARED REAL')
sg('send_command', 'init_distance TRAIN');
sg('send_command', 'new_classifier HIERARCHICAL');
sg('send_command', sprintf('train_classifier %d',merges));

[assignments,pairs]=sg('get_classifier');

%figure()
%clf
%plot(traindat(1,trainlab==+1), traindat(2,trainlab==+1),'rx');
%hold on
%plot(traindat(1,trainlab==+2), traindat(2,trainlab==+2),'bx');
%plot(traindat(1,trainlab==+3), traindat(2,trainlab==+3),'gx');
%plot(traindat(1,trainlab==+4), traindat(2,trainlab==+4),'cx');
%
%plot(centers(1,:), centers(2,:), 'ko');
%
%for i=1:k
%	t = linspace(0, 2*pi, 100);
%	plot(radi(i)*cos(t)+centers(1,i),radi(i)*sin(t)+centers(2,i),'k-')
%end
%
%
