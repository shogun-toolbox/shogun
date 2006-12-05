from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *

num_dat=200
width=5
dist = 2 ;

# positive examples
feat_pos=randn(2,num_dat)+dist ;
plot(feat_pos[0,:], feat_pos[1,:], ".") ;

# negative examples
feat_neg=randn(2,num_dat)-dist ;
plot(feat_neg[0,:], feat_neg[1,:], "r.") ;

# train svm
features = array((feat_neg, feat_pos)) ;
labels = array((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
feat = RealFeatures(concatenate(features,axis=1))
lab = Labels(concatenate(labels))
gk=GaussianKernel(feat,feat, width)
svm = SVMLight(10.0, gk, lab)
svm.train()

# compute output plot iso-lines
x1=linspace(-5,5, 100)
x2=linspace(-5,5, 100)
x,y=meshgrid(x1,x2);
feat_test=RealFeatures(array((ravel(x), ravel(y))))
gk.init(feat, feat_test, True)
z = svm.classify().get_labels().reshape((100,100))

pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
show()

