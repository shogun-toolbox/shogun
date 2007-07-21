%{
 #include "kernel/WeightedCommWordStringKernel.h" 
%}

%rename(WeightedCommWordStringKernel) CWeightedCommWordStringKernel;

%include "kernel/StringKernel.i" 
%include "kernel/WeightedCommWordStringKernel.h" 
