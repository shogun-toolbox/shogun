%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<DREAL>;
%template(WordKernel) CSimpleKernel<WORD>;
%template(CharKernel) CSimpleKernel<CHAR>;
%template(ByteKernel) CSimpleKernel<BYTE>;
%template(IntKernel) CSimpleKernel<INT>;
%template(ShortKernel) CSimpleKernel<SHORT>;
%template(UlongKernel) CSimpleKernel<ULONG>;
