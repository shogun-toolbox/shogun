%module SimpleKernel
%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/Kernel.i"
%include "lib/common.i"

%feature("notabstract") SimpleKernel;

%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<DREAL>;
%template(CharKernel) CSimpleKernel<CHAR>;
%template(IntKernel) CSimpleKernel<INT>;
