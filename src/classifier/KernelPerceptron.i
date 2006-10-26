%{
 #include "classifier/KernelPerceptron.h" 
%}

%rename(KernelPerceptron) CKernelPerceptron;

%include "kernel/KernelMachine.i" 
%include "classifier/KernelPerceptron.h" 
