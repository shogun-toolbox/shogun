%{
 #include "kernel/StringKernel.h" 
%}

%include "kernel/StringKernel.h" 

%template(StringRealKernel) CStringKernel<DREAL>;
%template(StringWordKernel) CStringKernel<uint16_t>;
%template(StringCharKernel) CStringKernel<char>;
%template(StringIntKernel) CStringKernel<int32_t>;
%template(StringUlongKernel) CStringKernel<uint64_t>;
%template(StringShortKernel) CStringKernel<SHORT>;
%template(StringByteKernel) CStringKernel<uint8_t>;
