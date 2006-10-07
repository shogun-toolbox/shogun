%module SimpleKernel
%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/Kernel.i"
%include "lib/common.i"

%feature("notabstract") SimpleKernel;

%include "kernel/SimpleKernel.h"

%template(CSimpleRealKernel) CSimpleKernel<DREAL>;
%template(CSimpleCharKernel) CSimpleKernel<CHAR>;
%template(CSimpleIntKernel) CSimpleKernel<INT>;
