import kernel.Kernel as k
import features.Features as f
import classifier.svm.SVM as svm
import numpy

dA = k.doubleArray(5)
dA[0] = 0.167
dA[1] = 0.25
dA[2] = 0.5
dA[3] = 0.25
dA[4] = 0.167

cA = numpy.chararray((12,1),1)
cA[0] = 'A'
cA[1] = 'A'
cA[2] = 'A'
cA[3] = 'A'
cA[4] = 'A'
cA[5] = 'A'
cA[6] = 'A'
cA[7] = 'A'
cA[8] = 'A'
cA[9] = 'A'
cA[10] = 'A'
cA[11] = 'A'

#f=cf.CCharFeatures(cf.DNA,'ACGTAAACCGGT',4,3)
feat123=f.CCharFeatures(f.DNA,4)
feat123.testNumpy(cA)
print
print 'test add param...'
x=numpy.zeros((4,3))
#print 'test add param2...'
#feat123.testAddParam2(1,x,4,3)
print 'test add param3...'
feat123.testAddParam3(x,4,3,1)
print 'test add param...'
feat123.testAddParam(1,cA,4,3)

print
print 'testing constructor...'
feat123=f.CCharFeatures(f.DNA,cA,4,3)
kernel = k.CWeightedDegreeCharKernel(feat123,feat123,10,dA,2,0,True,False,1)

trainlabels = f.CLabels(3)
trainlabels.set_int_label(0,1)
trainlabels.set_int_label(1,-1)
trainlabels.set_int_label(2,-1)

s = svm.CSVMLight()
s.set_labels(trainlabels)
s.set_kernel(kernel)
result = s.train();
print result
