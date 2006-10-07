%module RealKernel%{
 #include "kernel/RealKernel.h" 
 #include "kernel/SimpleKernel.h"
%}

%include "lib/common.i"
%include "kernel/SimpleKernel.i"

%feature("notabstract") RealKernel;

%include "kernel/RealKernel.h"
