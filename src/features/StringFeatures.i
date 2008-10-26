%{
 #include "features/StringFeatures.h"
%}

%include "lib/swig_typemaps.i"

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
#endif

%apply (ST** ARGOUT1, int32_t* DIM1) {(ST** dst, int32_t* len)};
%apply (char** ARGOUT1, int32_t* DIM1) {(char** dst, int32_t* len)};
%apply (uint8_t** ARGOUT1, int32_t* DIM1) {(uint8_t** dst, int32_t* len)};
%apply (SHORT** ARGOUT1, int32_t* DIM1) {(SHORT** dst, int32_t* len)};
%apply (uint16_t** ARGOUT1, int32_t* DIM1) {(uint16_t** dst, int32_t* len)};
%apply (int32_t** ARGOUT1, int32_t* DIM1) {(int32_t** dst, int32_t* len)};
%apply (uint32_t** ARGOUT1, int32_t* DIM1) {(uint32_t** dst, int32_t* len)};
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(int64_t** dst, int32_t* len)};
%apply (uint64_t** ARGOUT1, int32_t* DIM1) {(uint64_t** dst, int32_t* len)};

%include "features/StringFeatures.h"

/* workaround broken typemap %apply on templated classes */
%extend CStringFeatures<char>
{
    bool get_str(char** dst, int32_t* len)
    {
        self->CStringFeatures<char>::get_string(dst,len);
    }

    bool set_string_features(T_STRING<char>* strings, int32_t num_strings, int32_t max_len)
    {
        return self->CStringFeatures<char>::set_features(strings, num_strings, max_len);
    }
};
%extend CStringFeatures<uint8_t>
{
    void get_str(uint8_t** dst, int32_t* len)
    {
        self->CStringFeatures<uint8_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<SHORT>
{
    void get_str(SHORT** dst, int32_t* len)
    {
        self->CStringFeatures<SHORT>::get_string(dst,len);
    }
};
%extend CStringFeatures<uint16_t>
{
    void get_str(uint16_t** dst, int32_t* len)
    {
        self->CStringFeatures<uint16_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<int32_t>
{
    void get_str(int32_t** dst, int32_t* len)
    {
        self->CStringFeatures<int32_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<uint32_t>
{
    void get_str(uint32_t** dst, int32_t* len)
    {
        self->CStringFeatures<uint32_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<int64_t>
{
    void get_str(int64_t** dst, int32_t* len)
    {
        self->CStringFeatures<int64_t>::get_string(dst,len);
    }
};
%extend CStringFeatures<uint64_t>
{
    void get_str(uint64_t** dst, int32_t* len)
    {
        self->CStringFeatures<uint64_t>::get_string(dst,len);
    }
};

%template(StringCharFeatures) CStringFeatures<char>;
%template(StringByteFeatures) CStringFeatures<uint8_t>;
%template(StringShortFeatures) CStringFeatures<SHORT>;
%template(StringWordFeatures) CStringFeatures<uint16_t>;
%template(StringIntFeatures) CStringFeatures<int32_t>;
%template(StringUIntFeatures) CStringFeatures<uint32_t>;
%template(StringLongFeatures) CStringFeatures<int64_t>;
%template(StringUlongFeatures) CStringFeatures<uint64_t>;
