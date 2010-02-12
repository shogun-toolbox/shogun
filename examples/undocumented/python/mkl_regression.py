from sg import sg
from numpy import *

num=100
weight=1.
labels=concatenate((-ones([1,num]), ones([1,num])),1)[0]
features=concatenate((random.normal(size=(2,num))-1,random.normal(size=(2,num))+1),1)

tube_epsilon=1e-2

sg('new_classifier', 'MKL_REGRESSION')
sg('c', 1.)
sg('svr_tube_epsilon', tube_epsilon)

sg('set_labels', 'TRAIN', labels)
sg('add_features', 'TRAIN', features)
sg('add_features', 'TRAIN', features)
sg('add_features', 'TRAIN', features)

sg('set_kernel', 'COMBINED', 100)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 100.)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 10.)
sg('add_kernel', weight, 'GAUSSIAN', 'REAL', 100, 1.)
sg('train_classifier')
[bias, alphas]=sg('get_svm');
