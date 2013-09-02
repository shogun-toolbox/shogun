from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,legend, connect
from modshogun import *
from modshogun import *
from modshogun import *
import util

util.set_title('SVR on Sinus')

X, Y=util.get_sinedata()
C=10
width=0.5
epsilon=0.01

feat = RealFeatures(X)
lab = RegressionLabels(Y.flatten())
gk=GaussianKernel(feat,feat, width)
#svr = SVRLight(C, epsilon, gk, lab)
svr = LibSVR(C, epsilon, gk, lab)
svr.train()

plot(X, Y, '.', label='train data')
plot(X[0], svr.apply().get_labels(), hold=True, label='train output')

XE, YE=util.compute_output_plot_isolines_sine(svr, gk, feat, regression=True)
plot(XE[0], YE, hold=True, label='test output')

connect('key_press_event', util.quit)
show()
