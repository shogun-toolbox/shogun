from sg import sg
from pylab import figure,plot,show
from numpy import array,transpose,sin,double

sg('send_command', 'new_svm LIBSVR')
features=array([range(0,100)],dtype=double)
features.resize(1,100)
labels=sin(features)[0]
sg('set_features', "TRAIN", features)
sg('set_labels', "TRAIN", labels)
sg('send_command', 'set_kernel GAUSSIAN REAL 20 10')
sg('send_command', 'init_kernel TRAIN')
sg('send_command', 'c 1')
sg('send_command', 'svm_train')
[bias, alphas]=sg('get_svm');
sg('set_features', "TEST", features)
sg('send_command', 'init_kernel TEST')
out=sg('svm_classify');

figure()
plot(features[0],labels,'b-')
plot(features[0],labels,'bo')
plot(features[0],out,'r-')
plot(features[0],out,'ro')
show()
