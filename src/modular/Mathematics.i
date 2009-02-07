%{
 #include "lib/Mathematics.h" 
%}

%rename(Math) CMath;
%ignore RADIX_STACK_SIZE;

%include "lib/Mathematics.h" 
