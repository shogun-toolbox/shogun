modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');

% distance
disp('Distance')

feats_train=RealFeatures(fm_train_real);
feats_test=RealFeatures(fm_test_real);
width=1.7;
distance=EuclideanDistance();

kernel=DistanceKernel(feats_train, feats_test, width, distance);

km_train=kernel.get_kernel_matrix();
kernel.init(feats_train, feats_test);
km_test=kernel.get_kernel_matrix();

