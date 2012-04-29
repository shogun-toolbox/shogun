/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */
 
#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

/* These functions return new Objects */
%newobject get_transposed();

/* Remove C Prefix */
%rename(Alphabet) CAlphabet;
%rename(Features) CFeatures;
%rename(StreamingFeatures) CStreamingFeatures;
%rename(DotFeatures) CDotFeatures;
%rename(BinnedDotFeatures) CBinnedDotFeatures;
%rename(StreamingDotFeatures) CStreamingDotFeatures;
%rename(StreamingVwFeatures) CStreamingVwFeatures;
%rename(DummyFeatures) CDummyFeatures;
%rename(AttributeFeatures) CAttributeFeatures;
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
%rename(Subset) CSubset;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/FeatureTypes.h>
%include <shogun/features/Alphabet.h>
%include <shogun/lib/Compressor.h>
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>
%include <shogun/features/BinnedDotFeatures.h>
%include <shogun/features/StreamingFeatures.h>
%include <shogun/features/StreamingDotFeatures.h>
%include <shogun/features/StreamingVwFeatures.h>

/* Templated Class StringFeatures */
%include <shogun/features/StringFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StringBoolFeatures) CStringFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StringCharFeatures) CStringFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StringByteFeatures) CStringFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StringShortFeatures) CStringFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordFeatures) CStringFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringIntFeatures) CStringFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StringUIntFeatures) CStringFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StringLongFeatures) CStringFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongFeatures) CStringFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StringShortRealFeatures) CStringFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringRealFeatures) CStringFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StringLongRealFeatures) CStringFeatures<floatmax_t>;
#endif
}

/* Templated Class StreamingStringFeatures */
%include <shogun/features/StreamingStringFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingStringBoolFeatures) CStreamingStringFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingStringCharFeatures) CStreamingStringFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingStringByteFeatures) CStreamingStringFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingStringShortFeatures) CStreamingStringFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingStringWordFeatures) CStreamingStringFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingStringIntFeatures) CStreamingStringFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingStringUIntFeatures) CStreamingStringFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingStringLongFeatures) CStreamingStringFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingStringUlongFeatures) CStreamingStringFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingStringShortRealFeatures) CStreamingStringFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingStringRealFeatures) CStreamingStringFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingStringLongRealFeatures) CStreamingStringFeatures<floatmax_t>;
#endif
}

/* Templated Class StringFileFeatures */
%include <shogun/features/StringFileFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StringFileBoolFeatures) CStringFileFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StringFileCharFeatures) CStringFileFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StringFileByteFeatures) CStringFileFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StringFileShortFeatures) CStringFileFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StringFileWordFeatures) CStringFileFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringFileIntFeatures) CStringFileFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StringFileUIntFeatures) CStringFileFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StringFileLongFeatures) CStringFileFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StringFileUlongFeatures) CStringFileFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StringFileShortRealFeatures) CStringFileFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringFileRealFeatures) CStringFileFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StringFileLongRealFeatures) CStringFileFeatures<floatmax_t>;
#endif
}

/* Templated Class SparseFeatures */
%include <shogun/features/SparseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(SparseBoolFeatures) CSparseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(SparseCharFeatures) CSparseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(SparseByteFeatures) CSparseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(SparseShortFeatures) CSparseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(SparseWordFeatures) CSparseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(SparseIntFeatures) CSparseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(SparseUIntFeatures) CSparseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(SparseLongFeatures) CSparseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(SparseUlongFeatures) CSparseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(SparseShortRealFeatures) CSparseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(SparseRealFeatures) CSparseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(SparseLongRealFeatures) CSparseFeatures<floatmax_t>;
#endif
}

/* Templated Class StreamingSparseFeatures */
%include <shogun/features/StreamingSparseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingSparseBoolFeatures) CStreamingSparseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingSparseCharFeatures) CStreamingSparseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingSparseByteFeatures) CStreamingSparseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingSparseShortFeatures) CStreamingSparseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingSparseWordFeatures) CStreamingSparseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingSparseIntFeatures) CStreamingSparseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingSparseUIntFeatures) CStreamingSparseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingSparseLongFeatures) CStreamingSparseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingSparseUlongFeatures) CStreamingSparseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingSparseShortRealFeatures) CStreamingSparseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingSparseRealFeatures) CStreamingSparseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingSparseLongRealFeatures) CStreamingSparseFeatures<floatmax_t>;
#endif
}

/* Templated Class SimpleFeatures */
%include <shogun/features/SimpleFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(BoolFeatures) CSimpleFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(CharFeatures) CSimpleFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(ByteFeatures) CSimpleFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordFeatures) CSimpleFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortFeatures) CSimpleFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntFeatures)  CSimpleFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntFeatures)  CSimpleFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntFeatures)  CSimpleFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntFeatures)  CSimpleFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealFeatures) CSimpleFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealFeatures) CSimpleFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealFeatures) CSimpleFeatures<float64_t>;
#endif
}

/* Templated Class StreamingSimpleFeatures */
%include <shogun/features/StreamingSimpleFeatures.h>
namespace shogun
 {
#ifdef USE_BOOL
    %template(StreamingBoolFeatures) CStreamingSimpleFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingCharFeatures) CStreamingSimpleFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingByteFeatures) CStreamingSimpleFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingWordFeatures) CStreamingSimpleFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(StreamingShortFeatures) CStreamingSimpleFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingIntFeatures)  CStreamingSimpleFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingUIntFeatures)  CStreamingSimpleFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingLongIntFeatures)  CStreamingSimpleFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingULongIntFeatures)  CStreamingSimpleFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingLongRealFeatures) CStreamingSimpleFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingShortRealFeatures) CStreamingSimpleFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingRealFeatures) CStreamingSimpleFeatures<float64_t>;
#endif
}

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/AttributeFeatures.h>
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
%include <shogun/features/Subset.h>
