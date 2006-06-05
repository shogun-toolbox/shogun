import sg
import pylab
from numarray import *

sg.send_command('loglevel ALL')
sg.send_command('new_svm LIGHT')
features = array([[1.2, 3.5, -1], [1.0, 3.0, 0], [4, 5, 1], [1, 2, 3]])
labels = array([-1,1,-1,1])
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels)
sg.send_command('set_kernel GAUSSIAN REAL 20 10')
sg.send_command('init_kernel TRAIN')
km=sg.get_kernel_matrix()
sg.send_command('svm_train')
sv=sg.get_svm();
sg.set_features("TEST", features)
sg.send_command('init_kernel TEST')
out=sg.svm_classify();

pylab.figure()
pylab.scatter(features[:,0],features[:,1], s=80, marker='o', c=labels)
pylab.colorbar()
pylab.scatter(features[:,0],features[:,1], s=80, marker='o', c=labels)
pylab.show()



sg.send_command('new_svm SVRLIGHT')
features=array([arange(1,100)],Float64)
features.transpose()
labels=sin(features)
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels.flat)
sg.send_command('set_kernel GAUSSIAN REAL 20 10')
sg.send_command('init_kernel TRAIN')
sg.send_command('c 10')
sg.send_command('svm_train')
sv=sg.get_svm();
sg.set_features("TEST", features)
sg.send_command('init_kernel TEST')
light_out=sg.svm_classify();

sg.send_command('new_svm LIBSVR')
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels.flat)
sg.send_command('set_kernel GAUSSIAN REAL 20 10')
sg.send_command('c 0.1')
sg.send_command('init_kernel TRAIN')
sg.send_command('svm_train')
sv=sg.get_svm();
sg.set_features("TEST", features)
sg.send_command('init_kernel TEST')
libsvm_out=sg.svm_classify();

pylab.figure()
pylab.plot(features,labels,'b-')
pylab.plot(features,labels,'bo')
pylab.plot(features,libsvm_out,'r-')
pylab.plot(features,libsvm_out,'ro')
pylab.plot(features,light_out,'g-')
pylab.plot(features,light_out,'go')
pylab.show()
