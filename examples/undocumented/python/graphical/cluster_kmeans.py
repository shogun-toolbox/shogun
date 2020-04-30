import matplotlib.pyplot as plt
import numpy as np

import shogun as sg

k = 4
num = 1000
iter = 50000
dist = 2.2
traindat = np.concatenate((np.concatenate(
    (np.random.randn(1, num) - dist, np.random.randn(1, 2 * num) + dist, np.random.randn(1, num) + 2 * dist), 1),
                           np.concatenate((np.random.randn(1, num), np.random.randn(1, 2 * num) + dist,
                                           np.random.randn(1, num) - dist), 1)), 0)

trainlab = np.concatenate((np.ones(num), 2 * np.ones(num), 3 * np.ones(num), 4 * np.ones(num)))

feats_train = sg.create_features(traindat)
distance = sg.create_distance('EuclideanDistance')
distance.init(feats_train, feats_train)
kmeans = sg.create_machine('KMeans', k=k, distance=distance)
kmeans.train()

centers = kmeans.get('cluster_centers')
radi = kmeans.get('radiuses')

plt.figure()
plt.clf()
plt.plot(traindat[0, trainlab == +1], traindat[1, trainlab == +1], 'rx')
plt.plot(traindat[0, trainlab == +2], traindat[1, trainlab == +2], 'bx')
plt.plot(traindat[0, trainlab == +3], traindat[1, trainlab == +3], 'gx')
plt.plot(traindat[0, trainlab == +4], traindat[1, trainlab == +4], 'cx')

plt.plot(centers[0, :], centers[1, :], 'ko')

for i in range(k):
    t = np.linspace(0, 2 * np.pi, 100)
    plt.plot(radi[i] * np.cos(t) + centers[0, i], radi[i] * np.sin(t) + centers[1, i], 'k-')

plt.show()
