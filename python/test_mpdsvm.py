import sg
import pylab
from MLab import *
from numarray import *
import time

timing_value = 0.0

def tic():
        global timing_value
        timing_value = time.time()
        return timing_value

def toc():
        global timing_value
        if timing_value == 0:
                return None
        else:
                t = time.time() - timing_value
                timing_value = 0.0
                return t

num=1000;
sg.send_command('loglevel ALL')
#sg.send_command('new_svm LIGHT')
#sg.send_command('new_svm LIBSVM')
#sg.send_command('new_svm MPD')
#sg.send_command('new_svm GPBT')
features=concatenate((randn(num,2)-1,randn(num,2)+1),0)
labels=concatenate((-ones([1,num]), ones([1,num])),1).flat
sg.send_command('c 10')
#features = array([[1.2, 3.5, -1], [1.0, 3.0, 0], [4, 5, 1], [1, 2, 3],[2,2,2],[3,3,3],[4,4,4],[-1,0,1],[0,-1,1],[-1,-1,-1]])
#labels = array([ 1,1,-1,1,1,1,1,-1,-1,-1])
sg.set_features("TRAIN", features)
sg.set_labels("TRAIN", labels)
sg.send_command('set_kernel GAUSSIAN REAL 10 100')
sg.send_command('init_kernel TRAIN')
#km=sg.get_kernel_matrix()
#print km
tic()
#sg.send_command('svm_train')
#sv1=sg.get_svm();
print toc()
sg.send_command('new_svm LIGHT')
#sg.send_command('new_svm MPD')
sg.send_command('c 1000')
tic()
sg.send_command('svm_train')
print toc()
sv2=sg.get_svm();

#print sv1
print sv2
#sv=sg.get_svm();
#print sv
#
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
#pylab.savefig('bla.png', dpi=10, facecolor='w', edgecolor='w', orientation='portrait')
pylab.show()



#sg.send_command('new_svm SVRLIGHT')
#features=array([arange(1,100)],Float64)
#features.transpose()
#labels=sin(features)
#sg.set_features("TRAIN", features)
#sg.set_labels("TRAIN", labels.flat)
#sg.send_command('set_kernel GAUSSIAN REAL 20 10')
#sg.send_command('init_kernel TRAIN')
#sg.send_command('c 10')
#sg.send_command('svm_train')
#sv=sg.get_svm();
#sg.set_features("TEST", features)
#sg.send_command('init_kernel TEST')
#light_out=sg.svm_classify();
#
#sg.send_command('new_svm LIBSVR')
#sg.set_features("TRAIN", features)
#sg.set_labels("TRAIN", labels.flat)
#sg.send_command('set_kernel GAUSSIAN REAL 20 10')
#sg.send_command('c 0.1')
#sg.send_command('init_kernel TRAIN')
#sg.send_command('svm_train')
#sv=sg.get_svm();
#sg.set_features("TEST", features)
#sg.send_command('init_kernel TEST')
#libsvm_out=sg.svm_classify();
#
#pylab.figure()
#pylab.plot(features,labels,'b-')
#pylab.plot(features,labels,'bo')
#pylab.plot(features,libsvm_out,'r-')
#pylab.plot(features,libsvm_out,'ro')
#pylab.plot(features,light_out,'g-')
#pylab.plot(features,light_out,'go')
#pylab.show()
