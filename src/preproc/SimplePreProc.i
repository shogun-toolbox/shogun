%{
    #include "preproc/SimplePreProc.h" 
%}

%rename(SimplePreProc) CSimplePreProc;

%include "preproc/SimplePreProc.h"

%template(SimpleRealPreProc) CSimplePreProc<DREAL>;
%template(SimpleUlongPreProc) CSimplePreProc<ULONG>;
%template(SimpleWordPreProc) CSimplePreProc<WORD>;
%template(SimpleShortPreProc) CSimplePreProc<SHORT>;
%template(SimpleBytePreProc) CSimplePreProc<BYTE>;
%template(SimpleCharPreProc) CSimplePreProc<char>;
