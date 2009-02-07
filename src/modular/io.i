%{
 #include "lib/io.h" 
%}

%rename(IO) CIO;
%ignore NUM_LOG_LEVELS;
%ignore FBUFSIZE;

%include "lib/io.h" 
