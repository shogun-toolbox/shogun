from pylab import plot,grid,title,subplot,xlabel,ylabel,text,subplots_adjust,fill_between,mean,connect,show
from modshogun import GaussianKernel
from modshogun import LibSVM, LDA
from modshogun import ROCEvaluation
import util

util.set_title('ROC example')
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

# plot ROC for SVM
subplot(223)
ROC_evaluation=ROCEvaluation()
ROC_evaluation.evaluate(svm.apply(),labels)
roc = ROC_evaluation.get_ROC()
print roc
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
text(mean(roc[0])/2,mean(roc[1])/2,'auROC = %.5f' % ROC_evaluation.get_auROC())
grid(True)
xlabel('FPR')
ylabel('TPR')
title('LibSVM (Gaussian kernel, C=%.3f) ROC curve' % svm.get_C1(),size=10)

# plot ROC for LDA
subplot(224)
ROC_evaluation.evaluate(lda.apply(),labels)
roc = ROC_evaluation.get_ROC()
plot(roc[0], roc[1])
fill_between(roc[0],roc[1],0,alpha=0.1)
text(mean(roc[0])/2,mean(roc[1])/2,'auROC = %.5f' % ROC_evaluation.get_auROC())
grid(True)
xlabel('FPR')
ylabel('TPR')
title('LDA (gamma=%.3f) ROC curve' % lda.get_gamma(),size=10)

connect('key_press_event', util.quit)
show()

