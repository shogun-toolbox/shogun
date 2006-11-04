%{
    #include "kernel/WeightedDegreeCharKernel.h" 
%}

%include "lib/numpy.i"
%apply (DREAL* IN_ARRAY1, INT DIM1) {(DREAL* weights, INT d)};

%include "kernel/SimpleKernel.i"
%include "kernel/WeightedDegreeCharKernel.h" 

%pythoncode %{
  class WeightedDegreeCharKernel(CWeightedDegreeCharKernel):
     def __init__(self, features1, features2, size, type, degree, max_mismatch=0, use_normalization=True, block_computation=False, mkl_stepsize=1, which_degree=-1): 
        self.init(features1, features2, True)
        CWeightedDegreeCharKernel.__init__(self, size, type, len(w), max_mismatch, use_normalization, block_computation, mkl_stepsize, which_degree)
     def __init__(self, features1, features2, size, w, max_mismatch=0, use_normalization=True, block_computation=False, mkl_stepsize=1, which_degree=-1): 
        CWeightedDegreeCharKernel.__init__(self, size, E_EXTERNAL, len(w), max_mismatch, use_normalization, block_computation, mkl_stepsize, which_degree)
        self.set_wd_weights(w)
        self.init(features1, features2, True)
%}
