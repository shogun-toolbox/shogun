import sg
import pylab
from numarray import *

sg.send_command('new_svm LIBSVR')
features=array([arange(1,100)],Float64)
features.transpose()
labels=sin(features)
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels.flat)
sg.send_command('set_kernel GAUSSIAN REAL 20 10')
sg.send_command('init_kernel TRAIN')
sg.send_command('c 1')
sg.send_command('svm_train')
sv=sg.get_svm();
sg.set_features("TEST", features)
sg.send_command('init_kernel TEST')
out=sg.svm_classify();

pylab.figure()
pylab.plot(features,labels,'b-')
pylab.plot(features,labels,'bo')
pylab.plot(features,out,'r-')
pylab.plot(features,out,'ro')
pylab.show()
