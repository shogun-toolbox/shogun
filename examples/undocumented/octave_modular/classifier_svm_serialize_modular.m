% initialize modular shogun interface
modshogun

% add path to load matrix script
addpath('tools');

% some defines
C = 1.2;
width = 2.1;
epsilon = 1e-5;
num_threads = 2;

% get train features and labels
fm_train_real = load_matrix('../data/fm_train_real.dat');
fm_train_labels = load_matrix('../data/label_train_twoclass.dat');

% get test features and labels %fixme need example w/ test data/labels - using training data instead
fm_test_real = load_matrix('../data/fm_train_real.dat');
fm_test_labels = load_matrix('../data/label_train_twoclass.dat');

% create feature and label objects
feats_train = RealFeatures(fm_train_real);
feats_test = RealFeatures(fm_test_real);
labels = BinaryLabels(fm_train_labels);

% create kernel
kernel = GaussianKernel(feats_train, feats_train, width);

% create support vector machine
svm = LibSVM(C, kernel, labels);
svm.set_epsilon(epsilon);
svm.parallel.set_num_threads(num_threads);

% train
svm.train();

% save to file
file = SerializableAsciiFile('test_svm.dat', 'w');
svm.save_serializable(file);
file.close();

% load classifier and verify with test features
file_new = SerializableAsciiFile('test_svm.dat', 'r');
svm_new = LibSVM();
svm_new.load_serializable(file_new);
file_new.close();
result = svm_new.apply(feats_test).get_labels();

result = sum(sign(result) == fm_test_labels) / columns(fm_test_labels);
