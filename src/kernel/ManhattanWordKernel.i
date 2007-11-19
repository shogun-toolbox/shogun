%{
 #include "kernel/ManhattanWordKernel.h" 
%}

%rename(ManhattanWordKernel) CManhattanWordKernel;

%include "kernel/StringKernel.i" 
%include "kernel/ManhattanWordKernel.h" 
