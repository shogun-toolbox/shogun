from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double,zeros
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *

num_dat=1000
dist=0.5;
gamma=0.1;

# positive examples
feat_pos=randn(2,num_dat)+dist ;
plot(feat_pos[0,:], feat_pos[1,:], "r.") ;

# negative examples
feat_neg=randn(2,num_dat)-dist ;
plot(feat_neg[0,:], feat_neg[1,:], "b.") ;

# train lda
features = array((feat_neg, feat_pos)) ;
labels = array((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
feat = RealFeatures(concatenate(features,axis=1))
lab = Labels(concatenate(labels))
lda = LDA(gamma, feat, lab)
lda.train()

# compute output plot iso-lines
x1=linspace(-5,5, 100)
x2=linspace(-5,5, 100)
x,y=meshgrid(x1,x2);
feat_test=RealFeatures(array((ravel(x), ravel(y))))
lda.set_features(feat_test)
z = lda.classify().get_labels().reshape((100,100))

c=pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
colorbar(c)
show()

