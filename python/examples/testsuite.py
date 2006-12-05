#!/usr/bin/python

#
# A testsuite for the Kernels 
#

import unittest as ut

from shogun.Features import *
from shogun.Kernel import *
from shogun.Classifier import *

from numpy import *
from numpy.random import *

class TestKernels(ut.TestCase):
   """

   """
   
   def setUp(self):
      num_dat=50
      len=70
      acgt=array(['A','C','G','T'])
      trdat=chararray((len,2*num_dat),1,order='FORTRAN')
      trlab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
      for ix in xrange(2*num_dat):
          trdat[:,ix]=acgt[array(floor(4*random_sample(len)), dtype=int)]

      trdat[10:15,trlab==1]='A'
              
      self.trainfeat = CharFeatures(trdat,DNA)
      self.trainlab = Labels(trlab)

      simple2dData = array([[1.0,2,3,4,5,6,7,8,9,10],\
      [0.0,0,0,0,0,0,0,0,0,0]],order='FORTRAN')

      simple2dData.reshape((2,10))
      self.real_feat = RealFeatures(simple2dData)

   def testHierarchy(self):
      assert issubclass(RealKernel,CKernel)

      assert issubclass(AUCKernel,CKernel)
      assert issubclass(GaussianKernel,CKernel)
      assert issubclass(LinearKernel,CKernel)
      assert issubclass(PolyKernel,CKernel)
      
      assert issubclass(GaussianKernel,RealKernel)
      assert issubclass(LinearKernel,RealKernel)
      assert issubclass(PolyKernel,RealKernel)


   def testAUCKernel(self):
      #auck = AUCKernel(10,None)
      #auck.init()
      pass
   
   def testCanberraWordKernel(self):
      pass
   
   def testCharPolyKernel(self):
      pass
   
   def testChi2Kernel(self):
      #chi2k=Chi2Kernel(self.real_feat,self.real_feat, 1)
      #chi2k.get_kernel_type == K_CHI2
      #km = chi2k.get_kernel_matrix()
      #print km
	  pass
   
   def testCombinedKernel(self):
      pass
   
   def testCommUlongStringKernel(self):
      pass
   
   def testCommWordKernel(self):
      pass
   
   def testCommWordStringKernel(self):
      pass
   
   def testConstKernel(self):
      pass
   
   def testCustomKernel(self):
      pass
   
   def testDiagKernel(self):
      pass
   
   def testFixedDegreeCharKernel(self):
      pass
   
   def testGaussianKernel(self):
      gk=GaussianKernel(self.real_feat,self.real_feat, 1)
      km = gk.get_kernel_matrix()
      
   def testHammingWordKernel(self):
      pass
      
   def testHistogramWordKernel(self):
      pass
      
   def testKernelMachine(self):
      pass
      
   def testLinearByteKernel(self):
      pass
      
   def testLinearCharKernel(self):
      pass
      
   def testLinearKernel(self):
      lk = LinearKernel(self.real_feat,self.real_feat,1)
      lk.get_kernel_type == K_LINEAR
      km = lk.get_kernel_matrix()
      lk.cleanup()
      
   def testLinearWordKernel(self):
      pass
      
   def testLocalityImprovedCharKernel(self):
      pass
      
   def testManhattenWordKernel(self):
      pass
      
   def testMindyGramKernel(self):
      pass
      
   def testPolyKernel(self):
      pk = PolyKernel(self.real_feat,self.real_feat,1,2,True,True)
      pk.get_kernel_type = K_POLY
      km = pk.get_kernel_matrix()
      pk.cleanup()

      
   def testPolyMatchCharKernel(self):
      pass

   def testPolyMatchWordKernel(self):
      pass
      
   def testSalzbergWordKernel(self):
      pass
      
   def testSigmoidKernel(self):
      sk = SigmoidKernel(self.real_feat,self.real_feat,1,2.0,1.0)
      sk.get_kernel_type = K_SIGMOID
      km = sk.get_kernel_matrix()
      sk.cleanup()
      
   def testSimpleLocalityImprovedCharKernel(self):
      pass
      
   def testSparseGaussianKernel(self):
      pass
      
   def testSparseLinearKernel(self):
      pass
      
   def testSparseNormSquaredKernel(self):
      pass
      
   def testSparsePolyKernel(self):
      pass
      
   def testSpectrumKernel(self):
      pass
      
   def testWeightedDegreeCharKernel(self):
      pass
      
   def testWeightedDegreeCharKernelPolyA(self):
      pass
      
   def testWeightedDegreePositionCharKernel(self):
      pass
      
   def testWordMatchKernel(self):
      pass

   def tearDown(self):
      pass


class ClassifierTest(ut.TestCase):

   def setUp():
      pass

   def tearDown():
      pass


if __name__ == '__main__': 
   ut.main()
