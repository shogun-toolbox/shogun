import matplotlib.pyplot as plt
import shogun as sg
import util

plt.figure('KernelRidgeRegression on Sine')

X, Y = util.get_sinedata()
width = 1

feat = sg.features(X)
lab = sg.labels(Y.flatten())
gk = sg.kernel('GaussianKernel', log_width=width)
gk.init(feat, feat)
krr = sg.machine('KernelRidgeRegression', labels=lab, kernel=gk, tau=1e-3)
krr.train()

plt.scatter(X, Y,  label='train data', color='tab:red')
plt.plot(X[0], krr.apply().get('labels'), label='train output')

XE, YE = util.compute_output_plot_isolines_sine(krr, gk, feat, regression=True)
YE200 = krr.apply_one(200)

plt.plot(XE[0], YE, label='test output')
plt.plot([XE[0, 200]], [YE200], '+')

plt.legend()
plt.show()
