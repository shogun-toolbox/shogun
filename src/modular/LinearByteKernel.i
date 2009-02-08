%{
 #include <shogun/kernel/SimpleKernel.h>
 #include <shogun/kernel/LinearByteKernel.h>
%}

%rename(LinearByteKernel) CLinearByteKernel;

%include <shogun/kernel/LinearByteKernel.h>
