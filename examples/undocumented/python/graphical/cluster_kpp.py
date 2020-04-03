"""Graphical example illustrating improvement of convergence of KMeans
when cluster centers are initialized by KMeans++ algorithm.

In this example, 4 vertices of a rectangle are chosen: (0,0) (0,100) (10,0) (10,100).
There are 500 points normally distributed about each vertex.
Therefore, the ideal cluster centers for k=2 are the global minima ie (5,0) (5,100).

Written (W) 2014 Parijat Mazumdar
"""
import matplotlib.pyplot as plt
import numpy as np
import shogun as sg

k = 2
num = 500
d1 = np.concatenate((np.random.randn(1, num), 10. * np.random.randn(1, num)), 0)
d2 = np.concatenate((np.random.randn(1, num), 10. * np.random.randn(1, num)), 0) + np.array([[10.], [0.]])
d3 = np.concatenate((np.random.randn(1, num), 10. * np.random.randn(1, num)), 0) + np.array([[0.], [100.]])
d4 = np.concatenate((np.random.randn(1, num), 10. * np.random.randn(1, num)), 0) + np.array([[10.], [100.]])

traindata = np.concatenate((d1, d2, d3, d4), 1)
feat_train = sg.features(traindata)
distance = sg.distance('EuclideanDistance')
distance.init(feat_train, feat_train)

kmeans = sg.machine('KMeans', k=k, distance=distance, kmeanspp=True)
kmeans.train()
centerspp = kmeans.get('cluster_centers')
radipp = kmeans.get('radiuses')

kmeans = sg.machine('KMeans', k=k, distance=distance)
kmeans.train()
centers = kmeans.get('cluster_centers')
radi = kmeans.get('radiuses')

plt.figure('KMeans with KMeans++')
plt.plot(d1[0], d1[1], 'rx')
plt.plot(d2[0], d2[1], 'bx')
plt.plot(d3[0], d3[1], 'gx')
plt.plot(d4[0], d4[1], 'cx')

plt.plot(centerspp[0, :], centerspp[1, :], 'ko')
for i in range(k):
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(radipp[i] * np.cos(t) + centerspp[0, i], radipp[i] * np.sin(t) + centerspp[1, i], 'k-')

plt.figure('KMeans without KMeans++')
plt.plot(d1[0], d1[1], 'rx')
plt.plot(d2[0], d2[1], 'bx')
plt.plot(d3[0], d3[1], 'gx')
plt.plot(d4[0], d4[1], 'cx')

plt.plot(centers[0, :], centers[1, :], 'ko')
for i in range(k):
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(radi[i] * np.cos(t) + centers[0, i], radi[i] * np.sin(t) + centers[1, i], 'k-')

plt.show()
