library(sg)

traindat = c("AGTAA", "CGCCC", "GGCGG", "TGTCT")
trainlab <- c(1,-1,-1,1)
testdat = c("AGCAA", "CCCCC", "GGGGG", "TGCTT")

order = 2
C = 1.0

sg('loglevel', 'ALL')
sg('use_linadd', TRUE)
sg('mkl_parameters', 1e-5, 0)
sg('svm_epsilon', 1e-4)
sg('clean_features', 'TRAIN')
sg('clean_kernel')
sg('set_features', 'TRAIN', traindat, 'DNA')
sg('convert', 'TRAIN', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1)
sg('add_preproc', 'SORTWORDSTRING')
sg('attach_preproc', 'TRAIN')
sg('set_labels', 'TRAIN', trainlab)
sg('new_classifier', 'SVMLIGHT')
sg('set_kernel', 'COMMSTRING', 'WORD', 10, TRUE, 'FULL')
sg('c', C)
km=sg('get_kernel_matrix', 'TRAIN')
sg('train_classifier')
svmAsList=sg('get_svm')

sg('set_features', 'TEST', testdat, 'DNA')
sg('convert', 'TEST', 'STRING', 'CHAR', 'STRING', 'WORD', order, order-1)
sg('attach_preproc', 'TEST')
sg('init_kernel_optimization')
valout=sg('classify')
