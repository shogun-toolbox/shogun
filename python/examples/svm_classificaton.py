import sg
import pylab
from MLab import *
from numarray import *

num=200;
sg.send_command('loglevel ALL')
features=concatenate((randn(num,2)-1,randn(num,2)+1),0)
labels=concatenate((-ones([1,num]), ones([1,num])),1).flat
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels)
sg.send_command('set_kernel GAUSSIAN REAL 10 5')
sg.send_command('init_kernel TRAIN')
sg.send_command('new_svm GPBT')
sg.send_command('c 100')
sg.send_command('svm_train')
sv=sg.get_svm();
print sv
pylab.figure()

x1=pylab.linspace(1.2*min(features)[0],1.2*max(features)[0], 50)
x2=pylab.linspace(1.2*min(features)[1],1.2*max(features)[1], 50)
x,y=pylab.meshgrid(x1,x2);
testfeatures=transpose(array((ravel(x), ravel(y))))

sg.set_features("TEST", testfeatures)
sg.send_command('init_kernel TEST')
out=sg.svm_classify();

z=reshape(out,(50,50))
pylab.pcolor(x, y, transpose(z), shading='flat')
pylab.scatter(features[:,0],features[:,1], s=20, marker='o', c=labels, hold=True)
pylab.contour(x, y, transpose(z), linewidths=1, colors='black', hold=True)
pylab.colorbar()
pylab.show()
