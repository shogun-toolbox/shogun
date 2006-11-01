 %module(directors="1") CKernelMachine
%{
 #include "kernel/KernelMachine.h"
%}

%include "lib/common.i"

%feature("director") CKernelMachine;

%include "kernel/KernelMachine.h"
