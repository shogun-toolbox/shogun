%{
 #include "regression/KRR.h" 
%}

%rename(KRR) CKRR;

%include "kernel/KernelMachine.i"
%include "regression/KRR.h" 
