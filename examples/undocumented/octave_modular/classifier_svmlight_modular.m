modshogun

% Explicit examples on how to use the different classifiers

addpath('tools');
label_train_dna=load_matrix('../data/label_train_dna.dat');
fm_train_dna=load_matrix('../data/fm_train_dna.dat');
fm_test_dna=load_matrix('../data/fm_test_dna.dat');

% svm light
try
	disp('SVMLight')

	feats_train=StringCharFeatures(DNA);
	feats_train.set_features(fm_train_dna);
	feats_test=StringCharFeatures(DNA);
	feats_test.set_features(fm_test_dna);
	degree=20;

	kernel=WeightedDegreeStringKernel(feats_train, feats_train, degree);

	C=1.2;
	epsilon=1e-5;
	num_threads=3;
	labels=BinaryLabels(label_train_dna);

	svm=SVMLight(C, kernel, labels);
	svm.set_epsilon(epsilon);
	svm.parallel.set_num_threads(num_threads);
	svm.train();

	kernel.init(feats_train, feats_test);
	svm.apply().get_labels();
catch
	disp('No support for SVMLight available.')
end
