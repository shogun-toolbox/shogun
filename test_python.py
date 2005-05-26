import gf
import pylab
from numarray import *

gf.send_command('loglevel ALL')
gf.send_command('new_svm LIGHT')
features = array([[1.2, 3.5, -1], [1.0, 3.0, 0], [4, 5, 1], [1, 2, 3]])
labels = array([-1,1,-1,1])
gf.set_features("TRAIN", features)
gf.set_labels("TRAIN", labels)
gf.send_command('set_kernel GAUSSIAN REAL 20 10')
gf.send_command('init_kernel TRAIN')
km=gf.get_kernel_matrix()
gf.send_command('svm_train')
sv=gf.get_svm();
gf.set_features("TEST", features)
gf.send_command('init_kernel TEST')
out=gf.svm_classify();

pylab.figure()
pylab.scatter(features[:,0],features[:,1], s=80, marker='o', c=labels)
pylab.colorbar()
pylab.scatter(features[:,0],features[:,1], s=80, marker='o', c=labels)
pylab.show()



gf.send_command('new_svm SVRLIGHT')
features=array([arange(1,100)],Float64)
features.transpose()
labels=sin(features)
gf.set_features("TRAIN", features)
gf.set_labels("TRAIN", labels.flat)
gf.send_command('set_kernel GAUSSIAN REAL 20 10')
gf.send_command('init_kernel TRAIN')
gf.send_command('c 10')
gf.send_command('svm_train')
sv=gf.get_svm();
gf.set_features("TEST", features)
gf.send_command('init_kernel TEST')
light_out=gf.svm_classify();

gf.send_command('new_svm LIBSVR')
gf.set_features("TRAIN", features)
gf.set_labels("TRAIN", labels.flat)
gf.send_command('set_kernel GAUSSIAN REAL 20 10')
gf.send_command('c 0.1')
gf.send_command('init_kernel TRAIN')
gf.send_command('svm_train')
sv=gf.get_svm();
gf.set_features("TEST", features)
gf.send_command('init_kernel TEST')
libsvm_out=gf.svm_classify();

pylab.figure()
pylab.plot(features,labels,'b-')
pylab.plot(features,labels,'bo')
pylab.plot(features,libsvm_out,'r-')
pylab.plot(features,libsvm_out,'ro')
pylab.plot(features,light_out,'g-')
pylab.plot(features,light_out,'go')
pylab.show()
