from numpy.random import rand
from numpy import floor
from shogun.Features import RealFeatures
from shogun.Kernel import *
import m_print


traindat = rand(2,3)
testdat = rand(2,10)
train_feat = RealFeatures(traindat)
test_feat  = RealFeatures(testdat)

def write_testcase(kernelname=None,fun_name=None, km_train=None,km_test=None, traindat=None, testdat=None, dict={} ):
	value_list = dict.values()
	value_str  = '_'.join([str(x) for x in value_list])
	value_str  = value_str.replace('.', '')
	mfile = open('../testcases/mfiles/'+kernelname+'_'+value_str+'.m', mode='w')

	if(not(traindat==None)):
		m_print.print_mat(traindat,mfile, mat_name='traindat')

	if(not(testdat==None)):
		m_print.print_mat(testdat,mfile, mat_name='testdat')

	if(not(km_train==None)):
		m_print.print_mat(km_train, mfile, mat_name='km_train')

	if(not(km_test==None)):
		m_print.print_mat(km_test,mfile,  mat_name='km_test') 

	if(not(testdat==None)):
		mfile.write("functionname = '"+fun_name+"'\n")

	if(not(kernelname==None)):
		mfile.write("kernelname = '"+ kernelname+"'\n")

	for key in dict.keys():
		mfile.write(key+'= %r \n'%dict[key])

	mfile.close()





#write mfile for GaussianKernel

gk=GaussianKernel(train_feat,train_feat, 1.3, 10)
km=gk.get_kernel_matrix()
gk.init(train_feat, test_feat)
test_km=gk.get_kernel_matrix()

write_testcase(kernelname='GaussianKernel',fun_name='test_gaussian_kernel', km_train=km ,km_test=test_km, traindat=traindat, testdat=testdat,dict={'size_':10, 'width_':1.3})


#write mfile for Linear Kernel


lk=LinearKernel(train_feat,train_feat, True)
km_train=lk.get_kernel_matrix()
lk.init(train_feat, test_feat)
km_test=lk.get_kernel_matrix()

write_testcase(kernelname='LinearKernel',fun_name='test_linear_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat,dict={'bool1':'True'})

#write mfile for Linear Kernel


lk=LinearKernel(train_feat,train_feat, False)
km_train=lk.get_kernel_matrix()
lk.init(train_feat, test_feat)
km_test=lk.get_kernel_matrix()

write_testcase(kernelname='LinearKernel',fun_name='test_linear_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat,dict={'bool1':'False'})



#write mfile for Chi2Kernel


k=Chi2Kernel(train_feat,train_feat, 10)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='Chi2Kernel',fun_name='test_chi2_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10})


#write mfile for SigmoidKernel


k=SigmoidKernel(train_feat,train_feat, 10, 1.1, 1.3)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='SigmoidKernel',fun_name='test_sigmoid_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10, 'gamma_':1.1, 'coef0':1.3})

#write mfile for SigmoidKernel


k=SigmoidKernel(train_feat,train_feat, 10, 0.5, 0.7)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='SigmoidKernel',fun_name='test_sigmoid_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10, 'gamma_':0.5, 'coef0':0.7})

#write mfile for PolyKernel


k=PolyKernel(train_feat,train_feat, 10, 3, True, True)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='PolyKernel',fun_name='test_poly_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10,'degree':3, 'inhom':'True', 'use_norm':'True'})

#write mfile for PolyKernel


k=PolyKernel(train_feat,train_feat, 10, 3, False, True)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='PolyKernel',fun_name='test_poly_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10,'degree':3, 'inhom':'False', 'use_norm':'True'})


#write mfile for PolyKernel


k=PolyKernel(train_feat,train_feat, 10, 3, True, False)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='PolyKernel',fun_name='test_poly_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10,'degree':3, 'inhom':'True', 'use_norm':'False'})


#write mfile for PolyKernel


k=PolyKernel(train_feat,train_feat, 10, 3, False, False)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='PolyKernel',fun_name='test_poly_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10,'degree':3, 'inhom':'False', 'use_norm':'False'})



#write mfile for PolyKernel datatype CHAR
acgt = ['A', 'C', 'G','T']
clist = list()
for x in range(50):
	clist.append(acgt[int(floor(rand()*4))])

str_test_data = ''.join(clist)
	

k=PolyKernel(train_feat,train_feat, 10, 3, True, True)
km_train=k.get_kernel_matrix()
k.init(train_feat, test_feat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='PolyKernel',fun_name='test_poly_kernel', km_train=km_train ,km_test=km_test, traindat=traindat, testdat=testdat, dict={'size_':10,'degree':3, 'inhom':'False', 'use_norm':'False'})