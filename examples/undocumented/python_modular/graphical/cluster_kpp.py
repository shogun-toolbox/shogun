"""Graphical example illustrating improvement of convergence of KMeans
when cluster centers are initialized by KMeans++ algorithm.

In this example, 4 vertices of a rectangle are chosen: (0,0) (0,100) (10,0) (10,100).
There are 500 points normally distributed about each vertex.
Therefore, the ideal cluster centers for k=2 are the global minima ie (5,0) (5,100).

Written (W) 2014 Parijat Mazumdar
"""
from pylab import figure,clf,plot,linspace,pi,show
from numpy import array,ones,zeros,cos,sin,concatenate
from numpy.random import randn

from modshogun import *

k=2
num=500
d1=concatenate((randn(1,num),10.*randn(1,num)),0)
d2=concatenate((randn(1,num),10.*randn(1,num)),0)+array([[10.],[0.]])
d3=concatenate((randn(1,num),10.*randn(1,num)),0)+array([[0.],[100.]])
d4=concatenate((randn(1,num),10.*randn(1,num)),0)+array([[10.],[100.]])

traindata=concatenate((d1,d2,d3,d4),1)
feat_train=RealFeatures(traindata)
distance=EuclideanDistance(feat_train,feat_train)

kmeans=KMeans(k, distance, True)
kmeans.train()
centerspp=kmeans.get_cluster_centers()
radipp=kmeans.get_radiuses()

kmeans.set_use_kmeanspp(False)
kmeans.train()
centers=kmeans.get_cluster_centers()
radi=kmeans.get_radiuses()

figure('KMeans with KMeans++')
clf()
plot(d1[0],d1[1],'rx')
plot(d2[0],d2[1],'bx',hold=True)
plot(d3[0],d3[1],'gx',hold=True)
plot(d4[0],d4[1],'cx',hold=True)

plot(centerspp[0,:], centerspp[1,:], 'ko',hold=True)
for i in xrange(k):
	t = linspace(0, 2*pi, 100)
	plot(radipp[i]*cos(t)+centerspp[0,i],radipp[i]*sin(t)+centerspp[1,i],'k-', hold=True)

figure('KMeans w/o KMeans++')
clf()
plot(d1[0],d1[1],'rx')
plot(d2[0],d2[1],'bx',hold=True)
plot(d3[0],d3[1],'gx',hold=True)
plot(d4[0],d4[1],'cx',hold=True)

plot(centers[0,:], centers[1,:], 'ko',hold=True)
for i in xrange(k):
	t = linspace(0, 2*pi, 100)
	plot(radi[i]*cos(t)+centers[0,i],radi[i]*sin(t)+centers[1,i],'k-', hold=True)

show()
