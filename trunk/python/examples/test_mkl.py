import gf
from MLab import *
from numarray import *
num=100;
gf.send_command('loglevel ALL')
labels=concatenate((-ones([1,num]), ones([1,num])),1)[0]
features=concatenate((randn(num,2)-1,randn(num,2)+1),0)

gf.send_command('c 10')
gf.send_command('new_svm LIGHT')
gf.send_command('use_mkl 1')

gf.set_labels("TRAIN", labels)
gf.add_features("TRAIN", features)
gf.add_features("TRAIN", features)
gf.add_features("TRAIN", features)

gf.send_command('set_kernel COMBINED ANY 100')
gf.send_command('add_kernel 1 GAUSSIAN REAL 100 100')
gf.send_command('add_kernel 1 GAUSSIAN REAL 100 10')
gf.send_command('add_kernel 1 GAUSSIAN REAL 100 1')
gf.send_command('init_kernel TRAIN')
gf.send_command('svm_train')
sv1=gf.get_svm();
