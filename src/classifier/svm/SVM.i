%module SVM
%{
#define SWIG_FILE_WITH_INIT
 #include "classifier/svm/SVM.h"
%}

%init %{
	  import_array();
%}

%include "lib/common.i"
%include "lib/numpy.i"

%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** alphas, INT* d1)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* alphas, INT d)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* svs, INT d)};

%include "kernel/KernelMachine.i"
%include "classifier/svm/SVM.h"

#ifdef USE_SVMLIGHT
%include "classifier/svm/SVM_light.i"
#endif

%include "classifier/svm/LibSVM.i"

%pythoncode %{
  class SVM(CSVM):
      def __init__(self, kernel, alphas, support_vectors, b):
          assert(len(alphas)==len(support_vectors))
          num_sv=len(alphas)
          CSVM.__init__(self, num_sv)
          self.set_alphas(alphas)
          self.set_support_vectors(support_vectors)
          self.set_kernel(kernel)
          self.set_bias(b)
%}
