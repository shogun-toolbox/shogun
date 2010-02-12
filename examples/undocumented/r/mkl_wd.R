library(sg)

acgt <- c("A","C","G","T")
LT=sign(rnorm(1000))
XT= array("",dim=c(100,1000))

for (i in 1:length(XT)) {
   XT[i] = acgt[ceiling(4 * (rnorm(1) %% 1))]
}

for (k in c(30,60,61)) {
   for (i in 1:length(XT[k,])) {
      if (LT[i] == 1) {
         XT[k,i] = "A"
      }
   }
}

idx=sample(c(1:1000))
XTE=XT[,idx[1:200]]
LTE=LT[idx[1:200]]
XT=XT[,idx[201:1000]]
LT=LT[idx[201:1000]]

center_idx = 50
degree=3
mismatch = 0
C=1

#sg('loglevel', 'ALL')
sg('use_linadd', TRUE)
sg('mkl_parameters', 1e-5, 1)
sg('svm_epsilon', 1e-6)
sg('clean_features', 'TRAIN')
sg('clean_kernel')
sg('new_classifier', 'MKL_CLASSIFICATION')

sg('set_labels', 'TRAIN', LT)
sg('set_features', 'TRAIN', XT, 'DNA')
sg('set_kernel', 'WEIGHTEDDEGREE', 'CHAR', 10, degree, mismatch, FALSE, 1)

sg('c', C)
sg('svm_train')

svmAsList=sg('get_svm')
beta=sg('get_subkernel_weights')

sg('init_kernel_optimization')

sg('clean_features', 'TEST')
sg('set_features', 'TEST', XTE, 'DNA')

output_xte = sg('classify')

w=sg('get_subkernel_weights')
err=mean(sign(output_xte)!=LTE)
