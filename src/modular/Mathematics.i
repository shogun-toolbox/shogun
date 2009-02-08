%{
 #include <shogun/lib/Mathematics.h>
%}

%rename(Math) CMath;
%ignore RADIX_STACK_SIZE;

%include <shogun/lib/Mathematics.h>
