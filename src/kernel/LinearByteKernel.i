%{
 #include "kernel/SimpleKernel.h"
 #include "kernel/LinearByteKernel.h"
%}

%rename(LinearByteKernel) CLinearByteKernel;

%include "kernel/LinearByteKernel.h"
