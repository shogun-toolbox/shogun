%{
 #include "kernel/KernelMachine.h" 
 #include "classifier/KernelPerceptron.h" 
%}

%rename(KernelPerceptron) CKernelPerceptron;

%include "classifier/KernelPerceptron.h" 
