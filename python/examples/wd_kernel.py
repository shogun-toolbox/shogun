import kernel.WeightedDegreeCharKernel as wd
import features.CharFeatures as cf
import features.Labels as lab
import classifier.svm.SVM_light as svm

dA = wd.doubleArray(5)
dA[0] = 0.167
dA[1] = 0.25
dA[2] = 0.5
dA[3] = 0.25
dA[4] = 0.167
f=cf.CCharFeatures(cf.DNA,'ACGTAAACCGGT',4,3)
kernel = wd.CWeightedDegreeCharKernel(f,f,10,dA,2,0,True,False,1)

trainlabels = lab.CLabels(3)
trainlabels.set_int_label(0,1)
trainlabels.set_int_label(1,-1)
trainlabels.set_int_label(2,-1)

s = svm.CSVMLight()
s.set_labels(trainlabels)
s.set_kernel(kernel)
result = s.train();
print result
