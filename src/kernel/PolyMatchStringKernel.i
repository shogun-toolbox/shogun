%{
 #include "kernel/PolyMatchStringKernel.h"
%}

%rename(PolyMatchCharKernel) CPolyMatchCharKernel;

%include "kernel/PolyMatchStringKernel.h"
