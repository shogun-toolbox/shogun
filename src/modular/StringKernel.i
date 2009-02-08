%{
 #include <shogun/kernel/StringKernel.h>
%}

%include <shogun/kernel/StringKernel.h>

%template(StringRealKernel) CStringKernel<float64_t>;
%template(StringWordKernel) CStringKernel<uint16_t>;
%template(StringCharKernel) CStringKernel<char>;
%template(StringIntKernel) CStringKernel<int32_t>;
%template(StringUlongKernel) CStringKernel<uint64_t>;
%template(StringShortKernel) CStringKernel<int16_t>;
%template(StringByteKernel) CStringKernel<uint8_t>;
