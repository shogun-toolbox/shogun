%{
 #include "kernel/SparseKernel.h" 
%}

%include "kernel/SparseKernel.h" 

%template(SparseRealKernel) CSparseKernel<DREAL>;
%template(SparseWordKernel) CSparseKernel<uint16_t>;
