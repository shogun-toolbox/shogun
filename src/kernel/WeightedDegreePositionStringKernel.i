%{
#include "lib/common.h"
#include "kernel/WeightedDegreePositionStringKernel.h" 
%}

%include "kernel/WeightedDegreePositionStringKernel.h" 

#ifdef HAVE_PYTHON
%pythoncode %{
  import numpy
  class WeightedDegreePositionStringKernel(CWeightedDegreePositionStringKernel):
      def __init__(self, features1, features2, degree, shifts,
              use_normalization=True, max_mismatch=0, mkl_stepsize=1, weights=None, cache_size=10): 
          if weights is not None:
              assert(len(weights) == degree)
          else:
              weights = numpy.arange(1,degree+1,dtype=numpy.double)[::-1]/numpy.sum(numpy.arange(1,degree+1,dtype=numpy.double))

          CWeightedDegreePositionStringKernel.__init__(self, cache_size, weights, degree, max_mismatch, shifts, len(shifts), use_normalization, mkl_stepsize)
          self.init(features1, features2)
%}
#endif
