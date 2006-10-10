%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<DREAL>;
%template(CharKernel) CSimpleKernel<CHAR>;
%template(IntKernel) CSimpleKernel<INT>;
