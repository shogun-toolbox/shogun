%{
    #include "kernel/WeightedDegreeCharKernel.h" 
%}

#ifdef HAVE_PYTHON
%include "lib/python_typemaps.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* p_weights, INT d)};
#endif

%include "kernel/SimpleKernel.i"
%include "kernel/WeightedDegreeCharKernel.h" 

#ifdef HAVE_PYTHON
%pythoncode %{
  class WeightedDegreeCharKernel(CWeightedDegreeCharKernel):

      def __init__(self, features1, features2, degree, max_mismatch=0, use_normalization=True, block_computation=False, mkl_stepsize=1, which_degree=-1, weights=None, cache_size=10): 
          if weights is not None:
              assert(len(weights) == degree)
              CWeightedDegreeCharKernel.__init__(self, cache_size, E_EXTERNAL, degree, max_mismatch, use_normalization, block_computation, mkl_stepsize, which_degree)
              self.set_wd_weights(weights)
          else:
              CWeightedDegreeCharKernel.__init__(self, cache_size, E_WD, degree, max_mismatch, use_normalization, block_computation, mkl_stepsize, which_degree)
          self.init(features1, features2)
%}
#endif
