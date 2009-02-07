%{
   #include "kernel/SimpleKernel.h" 
%}

%include "kernel/SimpleKernel.h"

%template(RealKernel) CSimpleKernel<float64_t>;
%template(ShortRealKernel) CSimpleKernel<float32_t>;
%template(WordKernel) CSimpleKernel<uint16_t>;
%template(CharKernel) CSimpleKernel<char>;
%template(ByteKernel) CSimpleKernel<uint8_t>;
%template(IntKernel) CSimpleKernel<int32_t>;
%template(ShortKernel) CSimpleKernel<int16_t>;
%template(UlongKernel) CSimpleKernel<uint64_t>;
