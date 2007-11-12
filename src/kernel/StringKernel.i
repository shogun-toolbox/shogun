%{
 #include "kernel/StringKernel.h" 
%}

%include "kernel/StringKernel.h" 

%template(StringRealKernel) CStringKernel<DREAL>;
%template(StringWordKernel) CStringKernel<WORD>;
%template(StringCharKernel) CStringKernel<CHAR>;
%template(StringIntKernel) CStringKernel<INT>;
%template(StringUlongKernel) CStringKernel<ULONG>;
%template(StringShortKernel) CStringKernel<SHORT>;
%template(StringByteKernel) CStringKernel<BYTE>;
