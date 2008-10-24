%{
 #include "features/StringFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
#endif

%apply (ST** ARGOUT1, INT* DIM1) {(ST** dst, INT* len)};
%apply (char** ARGOUT1, INT* DIM1) {(char** dst, INT* len)};
%apply (BYTE** ARGOUT1, INT* DIM1) {(BYTE** dst, INT* len)};
%apply (SHORT** ARGOUT1, INT* DIM1) {(SHORT** dst, INT* len)};
%apply (WORD** ARGOUT1, INT* DIM1) {(WORD** dst, INT* len)};
%apply (INT** ARGOUT1, INT* DIM1) {(INT** dst, INT* len)};
%apply (UINT** ARGOUT1, INT* DIM1) {(UINT** dst, INT* len)};
%apply (LONG** ARGOUT1, INT* DIM1) {(LONG** dst, INT* len)};
%apply (ULONG** ARGOUT1, INT* DIM1) {(ULONG** dst, INT* len)};

%include "features/StringFeatures.h"

/* workaround broken typemap %apply on templated classes */
%extend CStringFeatures<char>
{
    bool get_str(char** dst, INT* len)
    {
        self->CStringFeatures<char>::get_string(dst,len);
    }

    bool set_string_features(T_STRING<char>* strings, INT num_strings, INT max_len)
    {
        return self->CStringFeatures<char>::set_features(strings, num_strings, max_len);
    }
};
%extend CStringFeatures<BYTE>
{
    void get_str(BYTE** dst, INT* len)
    {
        self->CStringFeatures<BYTE>::get_string(dst,len);
    }
};
%extend CStringFeatures<SHORT>
{
    void get_str(SHORT** dst, INT* len)
    {
        self->CStringFeatures<SHORT>::get_string(dst,len);
    }
};
%extend CStringFeatures<WORD>
{
    void get_str(WORD** dst, INT* len)
    {
        self->CStringFeatures<WORD>::get_string(dst,len);
    }
};
%extend CStringFeatures<INT>
{
    void get_str(INT** dst, INT* len)
    {
        self->CStringFeatures<INT>::get_string(dst,len);
    }
};
%extend CStringFeatures<UINT>
{
    void get_str(UINT** dst, INT* len)
    {
        self->CStringFeatures<UINT>::get_string(dst,len);
    }
};
%extend CStringFeatures<LONG>
{
    void get_str(LONG** dst, INT* len)
    {
        self->CStringFeatures<LONG>::get_string(dst,len);
    }
};
%extend CStringFeatures<ULONG>
{
    void get_str(ULONG** dst, INT* len)
    {
        self->CStringFeatures<ULONG>::get_string(dst,len);
    }
};

%template(StringCharFeatures) CStringFeatures<char>;
%template(StringByteFeatures) CStringFeatures<BYTE>;
%template(StringShortFeatures) CStringFeatures<SHORT>;
%template(StringWordFeatures) CStringFeatures<WORD>;
%template(StringIntFeatures) CStringFeatures<INT>;
%template(StringUIntFeatures) CStringFeatures<UINT>;
%template(StringLongFeatures) CStringFeatures<LONG>;
%template(StringUlongFeatures) CStringFeatures<ULONG>;
