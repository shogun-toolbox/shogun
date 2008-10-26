%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<DREAL>;
%template(ShortRealKernel) CSimpleKernel<SHORTREAL>;
%template(WordKernel) CSimpleKernel<uint16_t>;
%template(CharKernel) CSimpleKernel<char>;
%template(ByteKernel) CSimpleKernel<uint8_t>;
%template(IntKernel) CSimpleKernel<int32_t>;
%template(ShortKernel) CSimpleKernel<SHORT>;
%template(UlongKernel) CSimpleKernel<uint64_t>;
