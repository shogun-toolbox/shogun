%{
    #include "kernel/SpectrumKernel.h" 
%}

%rename(SpectrumKernel) CSpectrumKernel;

%include "kernel/StringKernel.i" 
%include "kernel/SpectrumKernel.h"

