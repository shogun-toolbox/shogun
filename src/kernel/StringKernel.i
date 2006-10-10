%{
 #include "kernel/StringKernel.h" 
%}

%include "kernel/StringKernel.h" 

%template(RealKernel) CStringKernel<DREAL>;
%template(WordKernel) CStringKernel<WORD>;
%template(CharKernel) CStringKernel<CHAR>;
%template(IntKernel) CStringKernel<INT>;
