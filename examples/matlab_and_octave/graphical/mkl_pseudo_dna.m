%mkl / wd kernel params
order=7;
mismatch=0;
C=2;

%data gen params
numneg=2000;
numpos=2000;
num=numpos+numneg;
len=50;
positions=[10,30];
variability=0.1;


%generate some toy data
rand('state',453456);
acgt=['A','C','G','T'];
motif{1}=['G','A','T','T','A','C','A'];
motif{2}=['A','G','T','A','G','T','G'];

idx=round(3*rand(len,numpos))+1;
seqs=acgt(idx);

for i=1:length(positions),
	for j=1:size(seqs,2),
		idx=ceil(4*rand(1,length(motif{i})));
		motifrnd=acgt(idx);
		p=randperm(length(motif{i}));
		b=round((length(motif{i})-1)*variability)+1;
		mot=motif{i};
		if variability>0,
			mot(p(1:b))=motifrnd(p(1:b));
		end
		seqs(positions(i):(positions(i)+length(motif{i}))-1,j)=mot';
	end
end

idx=round(3*rand(len,numneg))+1;
XT=[acgt(idx) seqs];
LT=[-ones(1,numneg), ones(1,numpos)];

p=randperm(num);
XV=char(XT(:,p(1001:end)));
LV=LT(p(1001:end));
XT=char(XT(:,p(1:1000)));
LT=LT(p(1:1000));

%do mkl with wd kernel
sg('set_features', 'TRAIN', XT, 'DNA');
sg('set_labels', 'TRAIN', LT);

sg('new_classifier', 'MKL_CLASSIFICATION');
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', 124, order, mismatch, false);
sg('c', C);

beta0=sg('get_subkernel_weights');
beta=beta0;
beta=repmat(beta,1,size(XT,1));
beta=beta/sum(beta(:));
sg('set_subkernel_weights',beta);

% compute optimal alphas / betas
sg('train_classifier');

% compute test output and evaluate
sg('set_features', 'TEST', XV, 'DNA');
out = sg('classify');
betas=sg('get_subkernel_weights');
acc=mean(LV==sign(out));
fprintf('accuracy: %f\n', acc)
imagesc(betas)
