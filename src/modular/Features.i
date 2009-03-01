/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
%define DOCSTR
"The `Features` module gathers all Feature objects available in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Features

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
%include "Features_doxygen.i"
#endif

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

/* Include Module Definitions */
%include "SGBase.i"
%{
#include <shogun/features/Features.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/features/SimpleFeatures.h>
#include <shogun/features/DummyFeatures.h>
#include <shogun/features/AttributeFeatures.h>
#include <shogun/features/Alphabet.h>
#include <shogun/features/CharFeatures.h>
#include <shogun/features/ByteFeatures.h>
#include <shogun/features/ShortFeatures.h>
#include <shogun/features/WordFeatures.h>
#include <shogun/features/IntFeatures.h>
#include <shogun/features/RealFeatures.h>
#include <shogun/features/ShortRealFeatures.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/CombinedDotFeatures.h>
#include <shogun/features/Labels.h>
#include <shogun/features/RealFileFeatures.h>
#include <shogun/features/FKFeatures.h>
#include <shogun/features/TOPFeatures.h>
#include <shogun/features/WDFeatures.h>
#include <shogun/features/ExplicitSpecFeatures.h>
#include <shogun/features/ImplicitWeightedSpecFeatures.h>
%}

/* Typemaps */
%apply (ST** ARGOUT1, int32_t* DIM1) {(ST** dst, int32_t* len)};
%apply (char** ARGOUT1, int32_t* DIM1) {(char** dst, int32_t* len)};
%apply (uint8_t** ARGOUT1, int32_t* DIM1) {(uint8_t** dst, int32_t* len)};
%apply (int16_t** ARGOUT1, int32_t* DIM1) {(int16_t** dst, int32_t* len)};
%apply (uint16_t** ARGOUT1, int32_t* DIM1) {(uint16_t** dst, int32_t* len)};
%apply (int32_t** ARGOUT1, int32_t* DIM1) {(int32_t** dst, int32_t* len)};
%apply (uint32_t** ARGOUT1, int32_t* DIM1) {(uint32_t** dst, int32_t* len)};
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(int64_t** dst, int32_t* len)};
%apply (uint64_t** ARGOUT1, int32_t* DIM1) {(uint64_t** dst, int32_t* len)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** matrix, int32_t* d1, int32_t* d2)};
%apply (int32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(int32_t** dst, int32_t* d1, int32_t* d2)};
%apply (int64_t** ARGOUT1, int32_t* DIM1) {(int64_t** h, int32_t* len)};
%apply (char* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(char* src, int32_t num_feat, int32_t num_vec)};
%apply (uint8_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint8_t* src, int32_t num_feat, int32_t num_vec)};
%apply (int16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(int16_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint16_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(uint16_t* src, int32_t num_feat, int32_t num_vec)};
%apply (uint16_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(uint16_t** dst, int32_t* d1, int32_t* d2)};
%apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float64_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float64_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float64_t** dst, int32_t* d1, int32_t* d2)};
%apply (float32_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {(float32_t* src, int32_t num_feat, int32_t num_vec)};
%apply (float32_t** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {(float32_t** dst, int32_t* d1, int32_t* d2)};
%apply (float64_t* IN_ARRAY1, int32_t DIM1) {(float64_t* labels, int32_t len)};
%apply (float64_t** ARGOUT1, int32_t* DIM1) {(float64_t** labels, int32_t* len)};


/* Remove C Prefix */
%rename(Features) CFeatures;
%rename(DotFeatures) CDotFeatures;
%rename(DummyFeatures) CDummyFeatures;
%rename(AttributeFeatures) CAttributeFeatures;
%rename(Alphabet) CAlphabet;
%rename(CharFeatures) CCharFeatures;
%rename(ByteFeatures) CByteFeatures;
%rename(ShortFeatures) CShortFeatures;
%rename(WordFeatures) CWordFeatures;
%rename(RealFeatures) CRealFeatures;
%rename(ShortRealFeatures) CShortRealFeatures;
%rename(IntFeatures) CIntFeatures;
%rename(CombinedFeatures) CCombinedFeatures;
%rename(CombinedDotFeatures) CCombinedDotFeatures;
%rename(Labels) CLabels;
%rename(RealFileFeatures) CRealFileFeatures;
%rename(FKFeatures) CFKFeatures;
%rename(TOPFeatures) CTOPFeatures;
%rename(WDFeatures) CWDFeatures;
%rename(ExplicitSpecFeatures) CExplicitSpecFeatures;
%rename(ImplicitWeightedSpecFeatures) CImplicitWeightedSpecFeatures;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>

/* Templated Class StringFeatures */
%include <shogun/features/StringFeatures.h>
%template(StringCharFeatures) CStringFeatures<char>;
%template(StringByteFeatures) CStringFeatures<uint8_t>;
%template(StringShortFeatures) CStringFeatures<int16_t>;
%template(StringWordFeatures) CStringFeatures<uint16_t>;
%template(StringIntFeatures) CStringFeatures<int32_t>;
%template(StringUIntFeatures) CStringFeatures<uint32_t>;
%template(StringLongFeatures) CStringFeatures<int64_t>;
%template(StringUlongFeatures) CStringFeatures<uint64_t>;

/* Templated Class SparseFeatures */
%include <shogun/features/SparseFeatures.h>
%template(SparseCharFeatures) CSparseFeatures<char>;
%template(SparseByteFeatures) CSparseFeatures<uint8_t>;
%template(SparseShortFeatures) CSparseFeatures<int16_t>;
%template(SparseWordFeatures) CSparseFeatures<uint16_t>;
%template(SparseIntFeatures) CSparseFeatures<int32_t>;
%template(SparseUIntFeatures) CSparseFeatures<uint32_t>;
%template(SparseLongFeatures) CSparseFeatures<int64_t>;
%template(SparseUlongFeatures) CSparseFeatures<uint64_t>;
%template(SparseRealFeatures) CSparseFeatures<float64_t>;
%template(SparseShortRealFeatures) CSparseFeatures<float32_t>;
%template(SparseLongRealFeatures) CSparseFeatures<float128_t>;

/* Templated Class SimpleFeatures */
%include <shogun/features/SimpleFeatures.h>
%template(SimpleRealFeatures) CSimpleFeatures<float64_t>;
%template(SimpleShortRealFeatures) CSimpleFeatures<float32_t>;
%template(SimpleByteFeatures) CSimpleFeatures<uint8_t>;
%template(SimpleWordFeatures) CSimpleFeatures<uint16_t>;
%template(SimpleShortFeatures) CSimpleFeatures<int16_t>;
%template(SimpleCharFeatures) CSimpleFeatures<char>;
%template(SimpleIntFeatures)  CSimpleFeatures<int32_t>;

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/AttributeFeatures.h>
%include <shogun/features/Alphabet.h>
%include <shogun/features/CharFeatures.h>
%include <shogun/features/ByteFeatures.h>
%include <shogun/features/ShortFeatures.h>
%include <shogun/features/IntFeatures.h>
%include <shogun/features/WordFeatures.h>
%include <shogun/features/RealFeatures.h>
%include <shogun/features/ShortRealFeatures.h>
%include <shogun/features/CombinedFeatures.h>
%include <shogun/features/CombinedDotFeatures.h>

%include <shogun/features/Labels.h>
%include <shogun/features/RealFileFeatures.h>
%include <shogun/features/FKFeatures.h>
%include <shogun/features/TOPFeatures.h>
%include <shogun/features/WDFeatures.h>
%include <shogun/features/ExplicitSpecFeatures.h>
%include <shogun/features/ImplicitWeightedSpecFeatures.h>

/* Templated Class MindyGramFeatures */
#ifdef HAVE_MINDY
%{
#include <shogun/features/MindyGramFeatures.h>
%}

%rename(MindyGramFeatures) CMindyGramFeatures;

%include <shogun/features/MindyGramFeatures.h>
%template(import_from_char) CMindyGramFeatures::import_features<char>;
%template(import_from_byte) CMindyGramFeatures::import_features<uint8_t>;
#endif

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

    bool set_string_features(T_STRING<uint8_t>* strings, int32_t num_strings, int32_t max_len)
    {
        return self->CStringFeatures<uint8_t>::set_features(strings, num_strings, max_len);
    }
};
%extend CStringFeatures<int16_t>
{
    void get_str(int16_t** dst, int32_t* len)
    {
        self->CStringFeatures<int16_t>::get_string(dst,len);
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
