%{
#include "kernel/WeightedDegreePositionCharKernel.h" 
%}

%include "kernel/WeightedDegreePositionCharKernel.h" 

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_weights, INT d)};
#endif

/*
#ifdef HAVE_PYTHON
%pythoncode %{
  import numpy
  class WeightedDegreePositionCharKernel(CWeightedDegreePositionCharKernel):

      def __init__(self, features1, features2, degree, max_mismatch=0, shifts,
              use_normalization=True, mkl_stepsize=1, weights=None, cache_size=10): 
          if weights is not None:
              assert(len(weights) == degree)
          else:
              weights = numpy.arange(1,degree+1,dtype=numpy.double)[::-1]/
                  numpy.sum(numpy.arange(1,degree+1,dtype=numpy.double))
          else:
              CWeightedDegreePositionCharKernel.__init__(self, cache_size, E_WD, degree, max_mismatch, use_normalization, block_computation, mkl_stepsize, which_degree)
          self.init(features1, features2)

		CWeightedDegreePositionStringKernel(LONG size, DREAL* weights, INT degree, INT max_mismatch, 
				INT * shift, INT shift_len, bool use_norm=false,
				INT mkl_stepsize=1) ;
%}
#endif
*/
