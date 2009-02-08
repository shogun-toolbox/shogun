%{
    #include <shogun/preproc/SimplePreProc.h>
%}

%rename(SimplePreProc) CSimplePreProc;

%include <shogun/preproc/SimplePreProc.h>

%template(SimpleRealPreProc) CSimplePreProc<float64_t>;
%template(SimpleUlongPreProc) CSimplePreProc<uint64_t>;
%template(SimpleWordPreProc) CSimplePreProc<uint16_t>;
%template(SimpleShortPreProc) CSimplePreProc<int16_t>;
%template(SimpleBytePreProc) CSimplePreProc<uint8_t>;
%template(SimpleCharPreProc) CSimplePreProc<char>;
