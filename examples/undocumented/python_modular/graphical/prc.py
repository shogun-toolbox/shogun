from pylab import plot,grid,title,subplot,xlabel,ylabel,text,subplots_adjust,fill_between,mean,connect,show
from modshogun import GaussianKernel
from modshogun import LibSVM, LDA
from modshogun import PRCEvaluation
import util

util.set_title('PRC example')
util.DISTANCE=0.5
subplots_adjust(hspace=0.3)

pos=util.get_realdata(True)
neg=util.get_realdata(False)
features=util.get_realfeatures(pos, neg)
labels=util.get_labels()

# classifiers
gk=GaussianKernel(features, features, 1.0)
svm = LibSVM(1000.0, gk, labels)
svm.train()
lda=LDA(1,features,labels)
lda.train()

## plot points
subplot(211)
plot(pos[0,:], pos[1,:], "r.")
plot(neg[0,:], neg[1,:], "b.")
grid(True)
title('Data',size=10)

# plot PRC for SVM
subplot(223)
PRC_evaluation=PRCEvaluation()
PRC_evaluation.evaluate(svm.apply(),labels)
PRC = PRC_evaluation.get_PRC()
plot(PRC[0], PRC[1])
fill_between(PRC[0],PRC[1],0,alpha=0.1)
text(0.55,mean(PRC[1])/3,'auPRC = %.5f' % PRC_evaluation.get_auPRC())
grid(True)
xlabel('Precision')
ylabel('Recall')
title('LibSVM (Gaussian kernel, C=%.3f) PRC curve' % svm.get_C1(),size=10)

# plot PRC for LDA
subplot(224)
PRC_evaluation.evaluate(lda.apply(),labels)
PRC = PRC_evaluation.get_PRC()
plot(PRC[0], PRC[1])
fill_between(PRC[0],PRC[1],0,alpha=0.1)
text(0.55,mean(PRC[1])/3,'auPRC = %.5f' % PRC_evaluation.get_auPRC())
grid(True)
xlabel('Precision')
ylabel('Recall')
title('LDA (gamma=%.3f) PRC curve' % lda.get_gamma(),size=10)

connect('key_press_event', util.quit)
show()

