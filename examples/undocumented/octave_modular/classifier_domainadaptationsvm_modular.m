addpath('tools');
modshogun;

tmp=load_matrix('../data/label_train_dna.dat');
label_train_dna=tmp(1:50);
tmp=load_matrix('../data/fm_train_dna.dat');
fm_train_dna=tmp(:,1:50);
tmp=load_matrix('../data/label_train_dna.dat');
label_train_dna2=tmp(50:92);
tmp=load_matrix('../data/fm_train_dna.dat');
fm_train_dna2=tmp(:, 50:92);
fm_test_dna=load_matrix('../data/fm_test_dna.dat');
fm_test_dna2=tmp(:,50:92);

%if exist('SVMLight')
	disp('Domain Adaptation SVM')

	C = 1.0;
	degree=3;

	feats_train=StringCharFeatures(DNA);
	feats_test=StringCharFeatures(DNA);
	feats_train.set_features(fm_train_dna);
	feats_test.set_features(fm_test_dna);
	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);
	labels=BinaryLabels(label_train_dna);
	svm=SVMLight(C, kernel, labels);
	svm.train();


	%#####################################

	disp('obtaining DA SVM from previously trained SVM')

	feats_train2=StringCharFeatures(DNA);
	feats_test2=StringCharFeatures(DNA);
	feats_train2.set_features(fm_train_dna2);
	feats_test2.set_features(fm_test_dna2);

	kernel2=WeightedDegreeStringKernel(feats_train, feats_train, degree);

	labels2=BinaryLabels(label_train_dna);

	% we regularize versus the previously obtained solution
	dasvm = DomainAdaptationSVM(C, kernel2, labels2, svm, 1.0);
	dasvm.train();

	out = dasvm.apply(feats_test2).get_labels();
%else
	%disp('No support for SVMLight available.')
%end
