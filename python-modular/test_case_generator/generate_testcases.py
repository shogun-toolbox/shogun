from numpy.random import *
from numpy import *
from shogun.PreProc import *
from shogun.Features import *
from shogun.Features import Alphabet,DNA
from shogun.Kernel import *
from shogun.Library import NO_NORMALIZATION
import m_print


traindat = rand(11,13)
testdat = rand(11,17)

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



#write mfile for WDKernel datatype CHAR
acgt  = array(['A', 'C', 'G','T'])
num_dat = 19
len = 11  
traindat = chararray((len,2*num_dat),1,order='FORTRAN')
for i in range(len):
        traindat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)), dtype=int)]

testdat = chararray((len,2*num_dat),1,order='FORTRAN')
for i in range(len):
        testdat[i,:]=acgt[array(floor(4*random_sample(2*num_dat)),dtype=int)]


#ctrain_feat = CharFeatures(traindat, DNA)
#ctest_feat  = CharFeatures(testdat,  DNA)


#k=WeightedDegreeCharKernel(ctrain_feat, ctrain_feat, 3)
#k.init(ctrain_feat, ctrain_feat)
#km_train=k.get_kernel_matrix()
#k.init(ctrain_feat, ctest_feat)
#km_test=k.get_kernel_matrix()

#write_testcase(kernelname='WeightedDegreeCharKernel',fun_name='test_wdchar_kernel', km_train=km_train ,km_test=km_test, traindat=matrix(traindat), testdat=matrix(testdat), dict={'alphabet':'DNA', 'size_':20,'degree':3, 'inhom':'True', 'use_norm':'True'})

#write mfile for WeightedDegreePositionStringKernel datatype STRING
seqlen=60;
degree=20;
acgt  = array(['A', 'C', 'G','T'])
trainlen = 11
testlen  = 17
traindat = []
testdat  = []
for i in range(trainlen):
	str1 = []
	str2 = []
	for j in range(seqlen):
		str1.append(acgt[floor(4*rand())])
		str2.append(acgt[floor(4*rand())])
        traindat.append(''.join(str1))
	testdat.append(''.join(str2))
	
for i in range(testlen-trainlen):
        str1 = []
	for j in range(seqlen):
                str1.append(acgt[floor(4*rand())])
	testdat.append(''.join(str1))

#alpha = Alphabet(DNA);
stringfeat = StringCharFeatures(DNA)
stringfeat.set_string_features(traindat)

stringtestfeat = StringCharFeatures(DNA)
stringtestfeat.set_string_features(testdat)

k= WeightedDegreePositionStringKernel(stringfeat, stringfeat, degree,  zeros(seqlen, dtype=int32))
km_train=k.get_kernel_matrix()
k.init(stringfeat, stringtestfeat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='WeightedDegreePositionStringKernel',fun_name='test_wdps_kernel', km_train=km_train ,km_test=km_test, traindat=matrix(traindat), testdat=matrix(testdat), dict={'alphabet':'DNA', 'degree':degree, 'seqlen':seqlen})

#write mfile for CommWordStringKernel datatype STRING
#--------------------------------------------------------------
reverse=False
gap=0
order=3

trainudat = StringWordFeatures(stringfeat.get_alphabet());
trainudat.obtain_from_char(stringfeat, order-1, order, gap, reverse)
testudat = StringWordFeatures(stringtestfeat.get_alphabet());
testudat.obtain_from_char(stringtestfeat, order-1, order, gap, reverse)


preproc = SortWordString();
preproc.init(trainudat);

trainudat.add_preproc(preproc)
trainudat.apply_preproc()

preproc = SortWordString();
preproc.init(testudat);

testudat.add_preproc(preproc)
testudat.apply_preproc()

k= CommWordStringKernel(trainudat,trainudat);# False, NO_NORMALIZATION, 10)
km_train=k.get_kernel_matrix()

k.init(trainudat,testudat)
km_test=k.get_kernel_matrix()

write_testcase(kernelname='CommWordStringKernel',fun_name='test_cws_kernel', km_train=km_train ,km_test=km_test, traindat=matrix(traindat), testdat=matrix(testdat), dict={'alphabet':'DNA', 'order':order, 'gap':gap, 'reverse':reverse})

