modshogun

addpath('tools');
label_train_multiclass=load_matrix('../data/label_train_multiclass.dat');
fm_train_real=load_matrix('../data/fm_train_real.dat');
fm_test_real=load_matrix('../data/fm_test_real.dat');


disp('Octave_modular')
% combined
disp('Combined')

kernel=CombinedKernel();
feats_train=CombinedFeatures();
feats_test=CombinedFeatures();

subkfeats_train=RealFeatures(fm_train_real);
subkfeats_test=RealFeatures(fm_test_real);
subkernel=GaussianKernel(10, 1.2);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);


subkfeats_train=RealFeatures(fm_train_real);
subkfeats_test=RealFeatures(fm_test_real);
subkernel=LinearKernel();
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);


subkfeats_train=RealFeatures(fm_train_real);
subkfeats_test=RealFeatures(fm_test_real);
subkernel=PolyKernel(10,2);
feats_train.append_feature_obj(subkfeats_train);
feats_test.append_feature_obj(subkfeats_test);
kernel.append_kernel(subkernel);
kernel.init(feats_train, feats_train);

C=1.2;
epsilon=1e-5;
num_threads=1;
labels=MulticlassLabels(label_train_multiclass);

% MKL_MULTICLASS
disp('MKL_MULTICLASS')
mkl=MKLMulticlass(C, kernel, labels);
mkl.set_epsilon(epsilon);
mkl.parallel.set_num_threads(num_threads);
mkl.set_mkl_epsilon(0.001);
mkl.set_mkl_norm(1.5);
mkl.train();

kernel.init(feats_train, feats_test);
result=mkl.apply().get_labels();
result
