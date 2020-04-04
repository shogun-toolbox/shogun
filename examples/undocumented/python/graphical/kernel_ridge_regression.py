import matplotlib.pyplot as plt
import shogun as sg
import util

plt.title('KernelRidgeRegression')

width = 2

# positive examples
pos = util.get_realdata(True)
plt.plot(pos[0, :], pos[1, :], 'r.')

# negative examples
neg = util.get_realdata(False)
plt.plot(neg[0, :], neg[1, :], 'b.')

# train krr
labels = util.get_labels()
train = util.get_realfeatures(pos, neg)
gk = sg.kernel('GaussianKernel', log_width=width)
gk.init(train, train)
krr = sg.machine('KernelRidgeRegression')
krr.put('labels', labels)
krr.put('kernel', gk)
krr.put('tau', 1e-3)
krr.train()

# compute output plot iso-lines
x, y, z = util.compute_output_plot_isolines(krr, gk, train, regression=True)

plt.pcolor(x, y, z)
plt.contour(x, y, z, linewidths=1, colors='black')

plt.show()
