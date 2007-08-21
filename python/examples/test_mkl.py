import sg
from MLab import *
from numpy import *
num=100;
sg.send_command('loglevel ALL')
labels=concatenate((-ones([1,num]), ones([1,num])),1)[0]
features=concatenate((randn(2,num)-1,randn(2,num)+1),1)

sg.send_command('c 10')
sg.send_command('new_svm LIGHT')
sg.send_command('use_mkl 1')

sg.set_labels("TRAIN", labels)
sg.add_features("TRAIN", features)
sg.add_features("TRAIN", features)
sg.add_features("TRAIN", features)

sg.send_command('set_kernel COMBINED 100')
sg.send_command('add_kernel 1 GAUSSIAN REAL 100 100')
sg.send_command('add_kernel 1 GAUSSIAN REAL 100 10')
sg.send_command('add_kernel 1 GAUSSIAN REAL 100 1')
sg.send_command('init_kernel TRAIN')
sg.send_command('svm_train')
sv1=sg.get_svm();
