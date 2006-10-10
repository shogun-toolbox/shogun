%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/Kernel.i"
%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<DREAL>;
%template(WordKernel) CSimpleKernel<WORD>;
%template(CharKernel) CSimpleKernel<CHAR>;
%template(IntKernel) CSimpleKernel<INT>;

