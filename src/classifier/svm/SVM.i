%{
 #include "classifier/svm/SVM.h"
%}

%include "lib/common.i"

%newobject classify;

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_support_vectors(self) -> [] of int") get_support_vectors;
%feature("autodoc", "get_alphas(self) -> [] of float") get_alphas;
#endif //HAVE_PYTHON

%apply (INT** ARGOUT1, INT* DIM1) {(INT** svs, INT* num)};
%apply (DREAL** ARGOUT1, INT* DIM1) {(DREAL** alphas, INT* d1)};
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* alphas, INT d)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* svs, INT d)};

#ifndef HAVE_PYTHON
%rename(SVM) CSVM;
#endif

%include "classifier/svm/SVM.h"

#ifdef USE_SVMLIGHT
%include "classifier/svm/SVM_light.i"
#endif //USE_SVMLIGHT


#ifdef HAVE_PYTHON
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
#endif //HAVE_PYTHON
