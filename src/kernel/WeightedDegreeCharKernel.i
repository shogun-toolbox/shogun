%{
    #include "kernel/WeightedDegreeCharKernel.h" 
%}

%include "lib/numpy.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* weights, INT d)};

%include "kernel/SimpleKernel.i"
%include "kernel/WeightedDegreeCharKernel.h" 

%pythoncode %{
  class WeightedDegreeCharKernel(CWeightedDegreeCharKernel):
     def __init__(self, features1, features2, size, w, max_mismatch=0, use_normalization=True, block_computation=True, mkl_stepsize=1): 
        CWeightedDegreeCharKernel.__init__(self, size, max_mismatch, use_normalization, block_computation, mkl_stepsize)
        self.init(features1, features2, True)
        self.set_wd_weights(w)
%}
