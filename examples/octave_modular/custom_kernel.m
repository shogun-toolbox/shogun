init_shogun;

C=1;
dim=7;

lab=sign(2*rand(1,dim) - 1);
data=rand(dim, dim);
symdata=data+data';

kernel=CustomKernel();
kernel.set_full_kernel_matrix_from_full(data);
labels=Labels(lab);
svm=LibSVM(C, kernel, labels);
svm.train();
out=svm.classify().get_labels();

