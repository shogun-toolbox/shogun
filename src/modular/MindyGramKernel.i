#ifdef HAVE_MINDY

%{
 #include <shogun/kernel/MindyGramKernel.h>
%}

%rename (MindyGramKernel) CMindyGramKernel;

%include <shogun/kernel/MindyGramKernel.h>
#endif
