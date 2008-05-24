from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,axis, connect
from numpy import array,meshgrid,reshape,linspace,ones,min,max
from numpy import concatenate,transpose,ravel,double,zeros
from numpy.random import randn
from shogun.Features import *
from shogun.Classifier import *
from shogun.Kernel import *
import util

util.set_title('SVM Linear 1')

num_dat=4000
distp=10
distn=10
C=1000

# positive examples
feat_pos=randn(2,num_dat)+distp

# negative examples
feat_neg=randn(2,num_dat)-distn

# train svm lin
features = array((feat_neg, feat_pos)) ;
labels = array((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
densefeat = RealFeatures(concatenate(features,axis=1))
feat=SparseRealFeatures()
feat.obtain_from_simple(densefeat)
lab = Labels(concatenate(labels))
svm = SVMLin(C, feat, lab)
svm.train()

lk=LinearKernel(densefeat,densefeat, 1.0)
svmlight = SVMLight(C, lk, lab)
svmlight.train()

x1_min=min(1.2*feat_neg[0,:])
x1_max=max(1.2*feat_pos[0,:])
x2_min=min(1.2*feat_neg[1,:])
x2_max=max(1.2*feat_pos[1,:])

# compute output plot iso-lines
x1=linspace(x1_min,x1_max, 100)
x2=linspace(x2_min,x2_max, 100)
x,y=meshgrid(x1,x2);
densefeat_test=RealFeatures(array((ravel(x), ravel(y))))
feat_test=SparseRealFeatures()
feat_test.obtain_from_simple(densefeat_test)
svm.set_features(feat_test)
z = svm.classify().get_labels().reshape((100,100))

lk.init(densefeat, densefeat_test)
zlight = svmlight.classify().get_labels().reshape((100,100))

c=pcolor(x, y, z, shading='interp')
contour(x, y, z, linewidths=1, colors='black', hold=True)
colorbar(c)
scatter(feat_pos[0,:], feat_pos[1,:],s=20, c='r')
scatter(feat_neg[0,:], feat_neg[1,:],s=20, c='b')
axis('tight')

figure()
util.set_title('SVM Linear 2')
c=pcolor(x, y, zlight, shading='interp')
contour(x, y, zlight, linewidths=1, colors='black', hold=True)
colorbar(c)

scatter(feat_neg[0,:], feat_neg[1,:],s=20, c='b')
scatter(feat_pos[0,:], feat_pos[1,:],s=20, c='r')
axis('tight')

connect('key_press_event', util.quit)
show()

