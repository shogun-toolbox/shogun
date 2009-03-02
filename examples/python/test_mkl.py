from sg import sg
from MLab import *
from numpy import *

num=100
weight=1.
labels=concatenate((-ones([1,num]), ones([1,num])),1)[0]
features=concatenate((randn(2,num)-1,randn(2,num)+1),1)

sg('c', 10.)
sg('new_svm', 'LIBSVM')
sg('use_mkl', True)

sg('set_labels', 'TRAIN', labels)
sg('add_features', 'TRAIN', features)
sg('add_features', 'TRAIN', features)
sg('add_features', 'TRAIN', features)

sg('set_kernel', 'COMBINED', 100)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 100.)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 10.)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 1.)
sg('init_kernel', 'TRAIN')
sg('train_classifier')
[bias, alphas]=sg('get_svm');
