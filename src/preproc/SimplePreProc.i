%{
    #include "preproc/SimplePreProc.h" 
%}

%rename(SimplePreProc) CSimplePreProc;

%include "preproc/SimplePreProc.h"

%template(SimpleRealPreProc) CSimplePreProc<DREAL>;
%template(SimpleUlongPreProc) CSimplePreProc<uint64_t>;
%template(SimpleWordPreProc) CSimplePreProc<uint16_t>;
%template(SimpleShortPreProc) CSimplePreProc<int16_t>;
%template(SimpleBytePreProc) CSimplePreProc<uint8_t>;
%template(SimpleCharPreProc) CSimplePreProc<char>;
