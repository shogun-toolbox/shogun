from sg import sg
from pylab import pcolor, scatter, contour, colorbar, show, imshow, connect
from numpy import min, max, where
import util

util.set_title('SVM Classification')

#sg('loglevel', 'ALL')
traindata=util.get_traindata()
labels=util.get_labels()
width=1.
size_cache=10

sg('set_features', 'TRAIN', traindata)
sg('set_labels', 'TRAIN', labels)
sg('set_kernel', 'GAUSSIAN', 'REAL', size_cache, width)
sg('new_classifier', 'LIBSVM')
sg('c', 100.)
sg('train_classifier')
[bias, alphas]=sg('get_svm')
#print bias
#print alphas

#print "objective: %f" % sg('get_svm_objective')

x, y=util.get_meshgrid(traindata)
testdata=util.get_testdata(x, y)
sg('set_features', 'TEST', testdata)
z=sg('classify')

z.resize((50,50))

#for non-smooth visualization
#pcolor(x, y, z, shading='flat')
#for smooth visualization
i=imshow(z,  origin='lower', extent=(
	1.2*min(traindata), 1.2*max(traindata), 1.2*min(traindata),
	1.2*max(traindata)))

negidx=where(labels==-1)[0]
posidx=where(labels==+1)[0]
scatter(traindata[0, negidx], traindata[1, negidx],
	s=20, marker='o', c='b', hold=True)
scatter(traindata[0, posidx], traindata[1, posidx],
	s=20, marker='o', c='r', hold=True)
contour(x, y, z, linewidths=1, colors='black', hold=True)
colorbar(i)
connect('key_press_event', util.quit)
show()
