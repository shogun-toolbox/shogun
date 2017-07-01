from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,legend,connect
from modshogun import *
import util

util.set_title('KernelRidgeRegression on Sine')


X, Y=util.get_sinedata()
width=1

feat=RealFeatures(X)
lab=RegressionLabels(Y.flatten())
gk=GaussianKernel(feat, feat, width)
krr=KernelRidgeRegression()
krr.set_labels(lab)
krr.set_kernel(gk)
krr.set_tau(1e-6)
krr.train()

plot(X, Y, '.', label='train data')
plot(X[0], krr.apply().get_labels(), hold=True, label='train output')

XE, YE=util.compute_output_plot_isolines_sine(krr, gk, feat, regression=True)
YE200=krr.apply_one(200)

plot(XE[0], YE, hold=True, label='test output')
plot([XE[0,200]], [YE200], '+', hold=True)
#print YE[200], YE200

connect('key_press_event', util.quit)
show()
