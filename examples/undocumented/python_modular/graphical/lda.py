from pylab import figure,pcolor,scatter,contour,colorbar,show,subplot,plot,connect
from modshogun import *
import util

util.set_title('LDA')
util.DISTANCE=0.5

gamma=0.1

# positive examples
pos=util.get_realdata(True)
plot(pos[0,:], pos[1,:], "r.")

# negative examples
neg=util.get_realdata(False)
plot(neg[0,:], neg[1,:], "b.")

# train lda
labels=util.get_labels()
features=util.get_realfeatures(pos, neg)
lda=LDA(gamma, features, labels)
lda.train()

# compute output plot iso-lines
x, y, z=util.compute_output_plot_isolines(lda)

c=pcolor(x, y, z)
contour(x, y, z, linewidths=1, colors='black', hold=True)
colorbar(c)

connect('key_press_event', util.quit)
show()
