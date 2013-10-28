modshogun

num=1000;
dist=1;
width=2.1;
C=1;

traindata_real=[randn(2,num)-dist, randn(2,num)+dist];
testdata_real=[randn(2,num)-dist, randn(2,num)+dist];

trainlab=[-ones(1,num), ones(1,num)];
testlab=[-ones(1,num), ones(1,num)];

feats_train=RealFeatures(traindata_real);
feats_test=RealFeatures(testdata_real);
kernel=GaussianKernel(feats_train, feats_train, width);

labels=BinaryLabels(trainlab);
svm=LibSVM(C, kernel, labels);
svm.parallel.set_num_threads(8);
svm.train();
kernel.init(feats_train, feats_test);
out=svm.apply().get_labels();
testerr=mean(sign(out)~=testlab)
