num=ceil(500*rand);
dist=10
traindat=[rand(2,num)-dist rand(2,num)+dist];
trainlab=[-ones(1,num) ones(1,num)];

sg('send_command', 'loglevel ALL');
sg('set_features', 'TRAIN', traindat);
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', 'set_distance NORMSQUARED REAL')
sg('send_command', 'init_distance TRAIN');
sg('send_command', 'new_classifier KMEANS');
sg('send_command', 'train_classifier 2');

[radi,centers]=sg('get_classifier');

plot(traindat(1,trainlab==+1), traindat(2,trainlab==+1),'ro');
plot(traindat(1,trainlab==-1), traindat(2,trainlab==-1),'bx');
colorbar

