%module(director="1") KernelMachine
%{
 #include "kernel/KernelMachine.h"
%}

%include "lib/common.i"

%feature("director") CKernelMachine;

%include "classifier/Classifier.i"
%include "kernel/KernelMachine.h"
