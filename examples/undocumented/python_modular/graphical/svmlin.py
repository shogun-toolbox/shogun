from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,axis, connect
from modshogun import *
import util

util.set_title('SVM Linear 1')
util.NUM_EXAMPLES=4000
C=1000

# positive examples
pos=util.get_realdata(True)
# negative examples
neg=util.get_realdata(False)

# train svm lin
labels=util.get_labels()
dense=util.get_realfeatures(pos, neg)
train=SparseRealFeatures()
train.obtain_from_simple(dense)
svm=SVMLin(C, train, labels)
svm.train()

lk=LinearKernel(dense, dense)
try:
	svmlight=LibSVM(C, lk, labels)
except NameError:
	print 'No SVMLight support available'
	import sys
	sys.exit(1)
svmlight.train()

x, y, z=util.compute_output_plot_isolines(svm, None, None, True, pos, neg)
x, y, zlight=util.compute_output_plot_isolines(svmlight, lk, dense, False, pos, neg)

c=pcolor(x, y, z)
contour(x, y, z, linewidths=1, colors='black', hold=True)
colorbar(c)
scatter(pos[0,:], pos[1,:],s=20, c='r')
scatter(neg[0,:], neg[1,:],s=20, c='b')
axis('tight')
connect('key_press_event', util.quit)

figure()
util.set_title('SVM Linear 2')
c=pcolor(x, y, zlight)
contour(x, y, zlight, linewidths=1, colors='black', hold=True)
colorbar(c)

scatter(neg[0,:], neg[1,:],s=20, c='b')
scatter(pos[0,:], pos[1,:],s=20, c='r')
axis('tight')
connect('key_press_event', util.quit)

show()
