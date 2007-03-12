%{
#include "lib/common.h"
#include "kernel/WeightedDegreePositionStringKernel.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY2, INT DIM1, INT DIM2) {(DREAL* weights, INT d, INT len)};
%apply (INT* IN_ARRAY1, INT DIM1) {(INT* shifts, INT len)};
#endif

%include "kernel/WeightedDegreePositionStringKernel.h" 

#ifdef HAVE_PYTHON
%pythoncode %{
  import numpy
  class WeightedDegreePositionStringKernel(CWeightedDegreePositionStringKernel):
      def __init__(self, features1, features2, degree, shifts,
              use_normalization=True, max_mismatch=0, mkl_stepsize=1, cache_size=10): 

          CWeightedDegreePositionStringKernel.__init__(self, cache_size, degree, max_mismatch, use_normalization, mkl_stepsize)
          self.set_shifts(shifts)
          if features1 is not None and features2 is not None:
              self.init(features1, features2)
%}
#endif
