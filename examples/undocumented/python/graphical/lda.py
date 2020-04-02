import matplotlib.pyplot as plt
from shogun import *
import util

plt.title('LDA')
util.DISTANCE = 0.5

gamma = 0.1

# positive examples
pos = util.get_realdata(True)
plt.plot(pos[0, :], pos[1, :], "r.")

# negative examples
neg = util.get_realdata(False)
plt.plot(neg[0, :], neg[1, :], "b.")

# train lda
labels = util.get_labels()
features = util.get_realfeatures(pos, neg)
lda = LDA(gamma, features, labels)
lda.train()

# compute output plot iso-lines
x, y, z = util.compute_output_plot_isolines(lda)

c = plt.pcolor(x, y, z)
plt.contour(x, y, z, linewidths=1, colors='black', hold=True)
plt.colorbar(c)

plt.show()
