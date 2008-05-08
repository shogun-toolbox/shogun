from sg import sg
from pylab import figure,plot,show
from numpy import array,transpose,sin,double

sg('new_regression', 'LIBSVR')
features=array([range(0,100)],dtype=double)
features.resize(1,100)
labels=sin(features)[0]
sg('set_features', "TRAIN", features)
sg('set_labels', "TRAIN", labels)
sg('set_kernel', 'GAUSSIAN', 'REAL', 20, 10.)
sg('init_kernel', 'TRAIN')
sg('c', 1.)
sg('train_regression')
[bias, alphas]=sg('get_svm');
sg('set_features', "TEST", features)
sg('init_kernel', 'TEST')
out=sg('classify');

figure()
plot(features[0],labels,'b-')
plot(features[0],labels,'bo')
plot(features[0],out,'r-')
plot(features[0],out,'ro')
show()
