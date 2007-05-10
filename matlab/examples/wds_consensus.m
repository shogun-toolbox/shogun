rand('seed',17);
%sequence lengths, number of sequences
len=100;
num_train=1000;

%SVM regularization factor C
C=1;

%Weighted Degree kernel parameters
order=20;
shift=15;
max_mismatch=0;
cache=10;
single_degree=-1;
x=shift*ones(1,len);
x(:)=0;
shifts = sprintf( '%i ', x(end:-1:1) );


%generate some toy data
acgt='ACGT';
traindat=[acgt(ceil(4*rand(len,num_train)))];
trainlab=[-ones(1,num_train/2),ones(1,num_train/2)];

traindat(10,trainlab==+1)='A';
traindat(11,trainlab==+1)='C';
traindat(12,trainlab==+1)='G';
traindat(13,trainlab==+1)='T';
traindat'
input('key to continue')

%train svm
sg('send_command', 'loglevel INFO' );
sg('set_features', 'TRAIN', traindat,'DNA');
sg('set_labels', 'TRAIN', trainlab);
sg('send_command', sprintf( 'set_kernel WEIGHTEDDEGREEPOS2 CHAR 10 %i %i %i %s', order, max_mismatch, len, shifts ) );
sg('send_command', 'init_kernel TRAIN');
sg('send_command', 'new_svm LIGHT');
sg('send_command', sprintf('c %f',C));
sg('send_command', 'svm_train');
consensus=sg('get_WD_consensus');
scores=sg('get_WD_scoring',1);
imagesc(reshape(scores,[4,length(scores)/4]))
consensus'

x=traindat(:,trainlab==1);
x(x=='A')=1;
x(x=='C')=2;
x(x=='G')=3;
x(x=='T')=4;
acgt(floor(median(x')))

sg('set_features', 'TEST', [ consensus  traindat(:,end)], 'DNA');
sg('send_command', 'init_kernel TEST');
[b,alphas]=sg('get_svm');
out=sg('svm_classify')-b;
out'
