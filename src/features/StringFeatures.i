%{
 #include "features/StringFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
#endif

%apply (ST** ARGOUT1, INT* DIM1) {(ST** dst, INT* len)};
%apply (char** ARGOUT1, INT* DIM1) {(char** dst, INT* len)};
%apply (uint8_t** ARGOUT1, INT* DIM1) {(uint8_t** dst, INT* len)};
%apply (SHORT** ARGOUT1, INT* DIM1) {(SHORT** dst, INT* len)};
%apply (uint16_t** ARGOUT1, INT* DIM1) {(uint16_t** dst, INT* len)};
%apply (INT** ARGOUT1, INT* DIM1) {(INT** dst, INT* len)};
%apply (uint32_t** ARGOUT1, INT* DIM1) {(uint32_t** dst, INT* len)};
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
%extend CStringFeatures<uint8_t>
{
    void get_str(uint8_t** dst, INT* len)
    {
        self->CStringFeatures<uint8_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<SHORT>
{
    void get_str(SHORT** dst, INT* len)
    {
        self->CStringFeatures<SHORT>::get_string(dst,len);
    }
};
%extend CStringFeatures<uint16_t>
{
    void get_str(uint16_t** dst, INT* len)
    {
        self->CStringFeatures<uint16_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<INT>
{
    void get_str(INT** dst, INT* len)
    {
        self->CStringFeatures<INT>::get_string(dst,len);
    }
};
%extend CStringFeatures<uint32_t>
{
    void get_str(uint32_t** dst, INT* len)
    {
        self->CStringFeatures<uint32_t>::get_string(dst,len);
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
%template(StringByteFeatures) CStringFeatures<uint8_t>;
%template(StringShortFeatures) CStringFeatures<SHORT>;
%template(StringWordFeatures) CStringFeatures<uint16_t>;
%template(StringIntFeatures) CStringFeatures<INT>;
%template(StringUIntFeatures) CStringFeatures<uint32_t>;
%template(StringLongFeatures) CStringFeatures<LONG>;
%template(StringUlongFeatures) CStringFeatures<ULONG>;
