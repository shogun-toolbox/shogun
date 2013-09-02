from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,connect,axis
from numpy.random import randn
from modshogun import *
from modshogun import *
from modshogun import *
import util

util.set_title('SVM')
util.NUM_EXAMPLES=200

width=5

# positive examples
pos=util.get_realdata(True)
plot(pos[0,:], pos[1,:], "r.")

# negative examples
neg=util.get_realdata(False)
plot(neg[0,:], neg[1,:], "b.")

# train svm
labels=util.get_labels()
train=util.get_realfeatures(pos, neg)
gk=GaussianKernel(train, train, width)
svm = LibSVM(10.0, gk, labels)
svm.train()

x, y, z=util.compute_output_plot_isolines(svm, gk, train)
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
axis('tight')

connect('key_press_event', util.quit)
show()

