from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double
from numpy.random import randn
from shogun.Features import *
from shogun.SVM import *
from shogun.Kernel import *

num_dat=200
width=0.5

feat = RealFeatures(concatenate((randn(2,num_dat)-1,randn(2,num_dat)+1),axis=1))
lab = Labels(concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double))))
gk=GaussianKernel(feat,feat, width)
svm = SVMLight(10.0, gk, lab)
svm.train()

x1=linspace(-5,5, 100)
x2=linspace(-5,5, 100)
x,y=meshgrid(x1,x2);
feat_test=RealFeatures(array((ravel(x), ravel(y))))
gk.init(feat, feat_test, True)
z = svm.classify().get_labels().reshape((100,100))
pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
show()
