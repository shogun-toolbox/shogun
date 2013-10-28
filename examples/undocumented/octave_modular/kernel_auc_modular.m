modshogun

addpath('tools');
fm_train_real=load_matrix('../data/fm_train_real.dat');
label_train_twoclass=load_matrix('../data/label_train_twoclass.dat');

% auc
disp('AUC')

feats_train=RealFeatures(fm_train_real);
width=1.7;
subkernel=GaussianKernel(feats_train, feats_train, width);

kernel=AUCKernel(0, subkernel);
kernel.setup_auc_maximization( BinaryLabels(label_train_twoclass) );
km_train=kernel.get_kernel_matrix();

