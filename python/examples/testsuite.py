#!/usr/bin/python

# Kernels 
try:
   import unittest as ut
   HAVE_UT = True
except:
   print '---=== You do not have a running unittest installation! ===---'
   import ut
   HAVE_UT = False

import shogun.Features as f
import shogun.Kernel as k
import shogun.SVM as s

class TestKernels(ut.TestCase):
   
   def setUp(self):
      trdat=chararray((len,2*num_dat),1,order='FORTRAN')
      trlab=concatenate((-ones(num_dat,dtype=double), ones(num_dat,dtype=double)))
      for ix in xrange(2*num_dat):
          trdat[:,ix]=acgt[array(floor(4*random_sample(len)), dtype=int)]

      trdat[10:15,trlab==1]='A'
              
      trainfeat = CharFeatures(trdat,DNA)
      trainlab = Labels(trlab)

   def testAUCKernel(self):
      auck = k.AUCKernel(10,None)
      auck.init()
      
      
      
      pass
   
   def testCanberraWordKernel(self):
      pass
   
   def testCharPolyKernel(self):
      pass
   
   def testChi2Kernel(self):
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
      pass
      
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
      pass
   def testLinearWordKernel(self):
      pass
   def testLocalityImprovedCharKernel(self):
      pass
   def testManhattenWordKernel(self):
      pass
   def testMindyGramKernel(self):
      pass
   def testPolyKernel(self):
      pass
   def testPolyMatchCharKernel(self):
      pass
   def testPolyMatchWordKernel(self):
      pass
   def testSalzbergWordKernel(self):
      pass
   def testSigmoidKernel(self):
      pass
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

if __name__ == '__main__': 
   try:
      unittest.main()
   except:
      t = TestKernels()
      t.run()
