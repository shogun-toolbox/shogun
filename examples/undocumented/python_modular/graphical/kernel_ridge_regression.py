from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,connect
from numpy import array,meshgrid,reshape,linspace,min,max
from numpy import concatenate,transpose,ravel
from modshogun import *
import util

util.set_title('KernelRidgeRegression')

width=20

# positive examples
pos=util.get_realdata(True)
plot(pos[0,:], pos[1,:], "r.")

# negative examples
neg=util.get_realdata(False)
plot(neg[0,:], neg[1,:], "b.")

# train krr
labels = util.get_labels(type='regression')
train = util.get_realfeatures(pos, neg)
gk = GaussianKernel(train, train, width)
krr = KernelRidgeRegression()
krr.set_labels(labels)
krr.set_kernel(gk)
krr.set_tau(1e-3)
krr.train()

# compute output plot iso-lines
x, y, z=util.compute_output_plot_isolines(krr, gk, train, regression=True)

pcolor(x, y, z)
contour(x, y, z, linewidths=1, colors='black', hold=True)

connect('key_press_event', util.quit)
show()
