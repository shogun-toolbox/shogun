%{
 #include "classifier/KRR.h" 
%}

%rename(KRR) CKRR;

%include "kernel/KernelMachine.i"
%include "classifier/KRR.h" 
