from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,legend,connect
from shogun.Features import *
from shogun.Regression import *
from shogun.Kernel import *
import util

util.set_title('KRR on Sine')


X, Y=util.get_sinedata()
width=1

feat=RealFeatures(X)
lab=Labels(Y.flatten())
gk=GaussianKernel(feat, feat, width)
krr=KRR()
krr.set_labels(lab)
krr.set_kernel(gk)
krr.set_tau(1e-6)
krr.train()

plot(X, Y, '.', label='train data')
plot(X[0], krr.classify().get_labels(), hold=True, label='train output')

XE, YE=util.compute_output_plot_isolines_sine(krr, gk, feat)
YE200=krr.classify_example(200)

plot(XE[0], YE, hold=True, label='test output')
plot([XE[0,200]], [YE200], '+', hold=True)
#print YE[200], YE200

connect('key_press_event', util.quit)
show()
