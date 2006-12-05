%{
 #include "kernel/SparseKernel.h" 
%}

%feature("notabstract") CKernel;

%include "kernel/Kernel.i" 
%include "kernel/SparseKernel.h" 

%template(SparseRealKernel) CSparseKernel<DREAL>;
%template(SparseWordKernel) CSparseKernel<WORD>;
