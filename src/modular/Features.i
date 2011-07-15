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
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Features_doxygen.i"
#endif
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
%include <shogun/features/FeatureTypes.h>
%include "Features_includes.i"
%include "Preprocessor_includes.i"
%include "Distribution_includes.i"
%include "Library_includes.i"
%include "Kernel_includes.i"
%include "Distance_includes.i"

%import "Preprocessor.i"
%import "Distribution.i"
%import "Library.i"

/* These functions return new Objects */
%newobject get_transposed();

/* Remove C Prefix */
%rename(BaseFeatures) CFeatures;
%rename(DotFeatures) CDotFeatures;
%rename(DummyFeatures) CDummyFeatures;
%rename(AttributeFeatures) CAttributeFeatures;
%rename(Alphabet) CAlphabet;
%rename(CombinedFeatures) CCombinedFeatures;
%rename(CombinedDotFeatures) CCombinedDotFeatures;
%rename(Labels) CLabels;
%rename(RealFileFeatures) CRealFileFeatures;
%rename(FKFeatures) CFKFeatures;
%rename(TOPFeatures) CTOPFeatures;
%rename(SNPFeatures) CSNPFeatures;
%rename(WDFeatures) CWDFeatures;
%rename(HashedWDFeatures) CHashedWDFeatures;
%rename(HashedWDFeaturesTransposed) CHashedWDFeaturesTransposed;
%rename(PolyFeatures) CPolyFeatures;
%rename(SparsePolyFeatures) CSparsePolyFeatures;
%rename(LBPPyrDotFeatures) CLBPPyrDotFeatures;
%rename(ExplicitSpecFeatures) CExplicitSpecFeatures;
%rename(ImplicitWeightedSpecFeatures) CImplicitWeightedSpecFeatures;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/lib/Compressor.h>
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>

/* Templated Class StringFeatures */
%include <shogun/features/StringFeatures.h>
namespace shogun
{
    %template(StringBoolFeatures) CStringFeatures<bool>;
    %template(StringCharFeatures) CStringFeatures<char>;
    %template(StringByteFeatures) CStringFeatures<uint8_t>;
    %template(StringShortFeatures) CStringFeatures<int16_t>;
    %template(StringWordFeatures) CStringFeatures<uint16_t>;
    %template(StringIntFeatures) CStringFeatures<int32_t>;
    %template(StringUIntFeatures) CStringFeatures<uint32_t>;
    %template(StringLongFeatures) CStringFeatures<int64_t>;
    %template(StringUlongFeatures) CStringFeatures<uint64_t>;
    %template(StringShortRealFeatures) CStringFeatures<float32_t>;
    %template(StringRealFeatures) CStringFeatures<float64_t>;
    %template(StringLongRealFeatures) CStringFeatures<floatmax_t>;
}

/* Templated Class StringFileFeatures */
%include <shogun/features/StringFileFeatures.h>
namespace shogun
{
    %template(StringFileBoolFeatures) CStringFileFeatures<bool>;
    %template(StringFileCharFeatures) CStringFileFeatures<char>;
    %template(StringFileByteFeatures) CStringFileFeatures<uint8_t>;
    %template(StringFileShortFeatures) CStringFileFeatures<int16_t>;
    %template(StringFileWordFeatures) CStringFileFeatures<uint16_t>;
    %template(StringFileIntFeatures) CStringFileFeatures<int32_t>;
    %template(StringFileUIntFeatures) CStringFileFeatures<uint32_t>;
    %template(StringFileLongFeatures) CStringFileFeatures<int64_t>;
    %template(StringFileUlongFeatures) CStringFileFeatures<uint64_t>;
    %template(StringFileShortRealFeatures) CStringFileFeatures<float32_t>;
    %template(StringFileRealFeatures) CStringFileFeatures<float64_t>;
    %template(StringFileLongRealFeatures) CStringFileFeatures<floatmax_t>;
}

/* Templated Class SparseFeatures */
%include <shogun/features/SparseFeatures.h>
namespace shogun
{
    %template(SparseBoolFeatures) CSparseFeatures<bool>;
    %template(SparseCharFeatures) CSparseFeatures<char>;
    %template(SparseByteFeatures) CSparseFeatures<uint8_t>;
    %template(SparseShortFeatures) CSparseFeatures<int16_t>;
    %template(SparseWordFeatures) CSparseFeatures<uint16_t>;
    %template(SparseIntFeatures) CSparseFeatures<int32_t>;
    %template(SparseUIntFeatures) CSparseFeatures<uint32_t>;
    %template(SparseLongFeatures) CSparseFeatures<int64_t>;
    %template(SparseUlongFeatures) CSparseFeatures<uint64_t>;
    %template(SparseShortRealFeatures) CSparseFeatures<float32_t>;
    %template(SparseRealFeatures) CSparseFeatures<float64_t>;
    %template(SparseLongRealFeatures) CSparseFeatures<floatmax_t>;
}

/* Templated Class SimpleFeatures */
%include <shogun/features/SimpleFeatures.h>
namespace shogun
{
    %template(BoolFeatures) CSimpleFeatures<bool>;
    %template(CharFeatures) CSimpleFeatures<char>;
    %template(ByteFeatures) CSimpleFeatures<uint8_t>;
    %template(WordFeatures) CSimpleFeatures<uint16_t>;
    %template(ShortFeatures) CSimpleFeatures<int16_t>;
    %template(IntFeatures)  CSimpleFeatures<int32_t>;
    %template(UIntFeatures)  CSimpleFeatures<uint32_t>;
    %template(LongIntFeatures)  CSimpleFeatures<int64_t>;
    %template(ULongIntFeatures)  CSimpleFeatures<uint64_t>;
    %template(LongRealFeatures) CSimpleFeatures<floatmax_t>;
    %template(ShortRealFeatures) CSimpleFeatures<float32_t>;
    %template(RealFeatures) CSimpleFeatures<float64_t>;
}

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/AttributeFeatures.h>
%include <shogun/features/Alphabet.h>
%include <shogun/features/CombinedFeatures.h>
%include <shogun/features/CombinedDotFeatures.h>

%include <shogun/features/Labels.h>
%include <shogun/features/RealFileFeatures.h>
%include <shogun/features/FKFeatures.h>
%include <shogun/features/TOPFeatures.h>
%include <shogun/features/SNPFeatures.h>
%include <shogun/features/WDFeatures.h>
%include <shogun/features/HashedWDFeatures.h>
%include <shogun/features/HashedWDFeaturesTransposed.h>
%include <shogun/features/PolyFeatures.h>
%include <shogun/features/SparsePolyFeatures.h>
%include <shogun/features/LBPPyrDotFeatures.h>
%include <shogun/features/ExplicitSpecFeatures.h>
%include <shogun/features/ImplicitWeightedSpecFeatures.h>
