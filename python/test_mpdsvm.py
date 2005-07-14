import gf
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

num=100;
gf.send_command('loglevel ALL')
#gf.send_command('new_svm LIGHT')
#gf.send_command('new_svm LIBSVM')
gf.send_command('new_svm MPD')
#gf.send_command('new_svm GPBT')
features=concatenate((randn(num,2)-1,randn(num,2)+1),0)
labels=concatenate((-ones([1,num]), ones([1,num])),1).flat
gf.send_command('c 10')
#features = array([[1.2, 3.5, -1], [1.0, 3.0, 0], [4, 5, 1], [1, 2, 3],[2,2,2],[3,3,3],[4,4,4],[-1,0,1],[0,-1,1],[-1,-1,-1]])
#labels = array([ 1,1,-1,1,1,1,1,-1,-1,-1])
gf.set_features("TRAIN", features)
gf.set_labels("TRAIN", labels)
gf.send_command('set_kernel GAUSSIAN REAL 100 100')
gf.send_command('init_kernel TRAIN')
#km=gf.get_kernel_matrix()
#print km
tic()
gf.send_command('svm_train')
sv1=gf.get_svm();
print toc()
gf.send_command('new_svm LIGHT')
#gf.send_command('new_svm MPD')
gf.send_command('c 1000')
tic()
gf.send_command('svm_train')
print toc()
sv2=gf.get_svm();

print sv1
print sv2
#sv=gf.get_svm();
#print sv
#
pylab.figure()

x1=pylab.linspace(1.2*min(features)[0],1.2*max(features)[0], 50)
x2=pylab.linspace(1.2*min(features)[1],1.2*max(features)[1], 50)
x,y=pylab.meshgrid(x1,x2);
testfeatures=transpose(array((ravel(x), ravel(y))))

gf.set_features("TEST", testfeatures)
gf.send_command('init_kernel TEST')
out=gf.svm_classify();

z=reshape(out,(50,50))

pylab.pcolor(x, y, transpose(z), shading='flat')
pylab.scatter(features[:,0],features[:,1], s=20, marker='o', c=labels, hold=True)
pylab.contour(x, y, transpose(z), linewidths=1, colors='black', hold=True)
pylab.colorbar()
pylab.show()



#gf.send_command('new_svm SVRLIGHT')
#features=array([arange(1,100)],Float64)
#features.transpose()
#labels=sin(features)
#gf.set_features("TRAIN", features)
#gf.set_labels("TRAIN", labels.flat)
#gf.send_command('set_kernel GAUSSIAN REAL 20 10')
#gf.send_command('init_kernel TRAIN')
#gf.send_command('c 10')
#gf.send_command('svm_train')
#sv=gf.get_svm();
#gf.set_features("TEST", features)
#gf.send_command('init_kernel TEST')
#light_out=gf.svm_classify();
#
#gf.send_command('new_svm LIBSVR')
#gf.set_features("TRAIN", features)
#gf.set_labels("TRAIN", labels.flat)
#gf.send_command('set_kernel GAUSSIAN REAL 20 10')
#gf.send_command('c 0.1')
#gf.send_command('init_kernel TRAIN')
#gf.send_command('svm_train')
#sv=gf.get_svm();
#gf.set_features("TEST", features)
#gf.send_command('init_kernel TEST')
#libsvm_out=gf.svm_classify();
#
#pylab.figure()
#pylab.plot(features,labels,'b-')
#pylab.plot(features,labels,'bo')
#pylab.plot(features,libsvm_out,'r-')
#pylab.plot(features,libsvm_out,'ro')
#pylab.plot(features,light_out,'g-')
#pylab.plot(features,light_out,'go')
#pylab.show()
