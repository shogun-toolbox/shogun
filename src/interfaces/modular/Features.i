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
%newobject create_merged_copy(CFeatures* other);
%newobject copy_subset(SGVector<index_t> indices);
%newobject get_streamed_features(index_t num_elements);


/* methods in the labels factory do a conversion only - but need to be ref'd for modular interfaces */
%newobject shogun::CLabelsFactory::to_binary(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_latent(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_multiclass(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_regression(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_structured(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_multilabel_output(CLabels* base_labels);
%newobject shogun::CLabelsFactory::to_multiclass_structured(CLabels* base_labels);

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::CDirectorDotFeatures;
%feature("director:except") {
    if ($error != NULL) {
        throw Swig::DirectorMethodException();
    }
}
#endif

#ifndef SWIGPYTHON
#define PROTOCOLS_DENSEFEATURES(class_name, type_name, format_str, typecode)
#define PROTOCOLS_DENSELABELS(class_type, class_name, type_name, format_str, typecode)
#define EXTEND_DENSEFEATURES(class_name, type_name, typecode)
#endif

/* Remove C Prefix */
%rename(Alphabet) CAlphabet;
%rename(Features) CFeatures;
%rename(StreamingFeatures) CStreamingFeatures;
%rename(DotFeatures) CDotFeatures;
%rename(DirectorDotFeatures) CDirectorDotFeatures;
%rename(BinnedDotFeatures) CBinnedDotFeatures;
%rename(StreamingDotFeatures) CStreamingDotFeatures;
%rename(StreamingVwFeatures) CStreamingVwFeatures;
%rename(DummyFeatures) CDummyFeatures;
%rename(IndexFeatures) CIndexFeatures;
%rename(AttributeFeatures) CAttributeFeatures;
%rename(CombinedFeatures) CCombinedFeatures;
%rename(CombinedDotFeatures) CCombinedDotFeatures;
%rename(HashedDocDotFeatures) CHashedDocDotFeatures;
%rename(StreamingHashedDocDotFeatures) CStreamingHashedDocDotFeatures;
%rename(RandomKitchenSinksDotFeatures) CRandomKitchenSinksDotFeatures;
%rename(RandomFourierDotFeatures) CRandomFourierDotFeatures;
%rename(Labels) CLabels;
%rename(LabelsFactory) CLabelsFactory;

PROTOCOLS_DENSELABELS(CDenseLabels, DenseLabels, float64_t, "d\0", NPY_FLOAT64)
%rename(DenseLabels) CDenseLabels;

PROTOCOLS_DENSELABELS(CBinaryLabels, BinaryLabels, float64_t, "d\0", NPY_FLOAT64)
%rename(BinaryLabels) CBinaryLabels;

PROTOCOLS_DENSELABELS(CMulticlassLabels, MulticlassLabels, float64_t, "d\0", NPY_FLOAT64)
%rename(MulticlassLabels) CMulticlassLabels;

PROTOCOLS_DENSELABELS(CRegressionLabels, RegressionLabels, float64_t, "d\0", NPY_FLOAT64)
%rename(RegressionLabels) CRegressionLabels;

%rename(StructuredLabels) CStructuredLabels;
%rename(LatentLabels) CLatentLabels;
%rename(MultilabelLabels) CMultilabelLabels;
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
%rename(DataGenerator) CDataGenerator;
%rename(LatentFeatures) CLatentFeatures;

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/FeatureTypes.h>
%include <shogun/features/Alphabet.h>
%include <shogun/lib/Compressor.h>
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>
%include <shogun/features/DirectorDotFeatures.h>
%include <shogun/features/BinnedDotFeatures.h>
%include <shogun/features/streaming/StreamingFeatures.h>
%include <shogun/features/streaming/StreamingDotFeatures.h>
%include <shogun/features/streaming/StreamingVwFeatures.h>
%include <shogun/features/DataGenerator.h>

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
%include <shogun/features/streaming/StreamingStringFeatures.h>
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
%include <shogun/features/streaming/StreamingSparseFeatures.h>
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

/* Templated Class DenseFeatures */
%include <shogun/features/DenseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
	// something wrong with git
	PROTOCOLS_DENSEFEATURES(BoolFeatures, bool, "?\0", NPY_BOOL)
	%template(BoolFeatures) CDenseFeatures<bool>;
	EXTEND_DENSEFEATURES(BoolFeatures, bool, NPY_BOOL)
#endif

#ifdef USE_CHAR
	PROTOCOLS_DENSEFEATURES(CharFeatures, char, "c\0", NPY_STRING)
	%template(CharFeatures) CDenseFeatures<char>;
	EXTEND_DENSEFEATURES(CharFeatures, char, NPY_STRING)
#endif

#ifdef USE_UINT8
	PROTOCOLS_DENSEFEATURES(ByteFeatures, uint8_t, "B\0", NPY_UINT8)
	%template(ByteFeatures) CDenseFeatures<uint8_t>;
	EXTEND_DENSEFEATURES(ByteFeatures, uint8_t, NPY_UINT8)
#endif

#ifdef USE_UINT16
	PROTOCOLS_DENSEFEATURES(WordFeatures, uint16_t, "H\0", NPY_UINT16)
	%template(WordFeatures) CDenseFeatures<uint16_t>;
	EXTEND_DENSEFEATURES(WordFeatures, uint16_t, NPY_UINT16)
#endif

#ifdef USE_INT16
	PROTOCOLS_DENSEFEATURES(ShortFeatures, int16_t, "h\0", NPY_INT16)
	%template(ShortFeatures) CDenseFeatures<int16_t>;
	EXTEND_DENSEFEATURES(ShortFeatures, int16_t, NPY_INT16)
#endif

#ifdef USE_INT32
	PROTOCOLS_DENSEFEATURES(IntFeatures, int32_t, "i\0", NPY_INT32)
	%template(IntFeatures)  CDenseFeatures<int32_t>;
	EXTEND_DENSEFEATURES(IntFeatures, int32_t, NPY_INT32)
#endif

#ifdef USE_UINT32
	PROTOCOLS_DENSEFEATURES(UIntFeatures, uint32_t, "I\0", NPY_UINT32)
	%template(UIntFeatures)  CDenseFeatures<uint32_t>;
	EXTEND_DENSEFEATURES(UIntFeatures, uint32_t, NPY_UINT32)
#endif

#ifdef USE_INT64
	PROTOCOLS_DENSEFEATURES(LongIntFeatures, int64_t, "l\0", NPY_INT64)
	%template(LongIntFeatures) CDenseFeatures<int64_t>;
	EXTEND_DENSEFEATURES(LongIntFeatures, int64_t, NPY_INT64)
#endif

#ifdef USE_UINT64
	PROTOCOLS_DENSEFEATURES(ULongIntFeatures, uint64_t, "L\0", NPY_UINT64)
	%template(ULongIntFeatures) CDenseFeatures<uint64_t>;
	EXTEND_DENSEFEATURES(ULongIntFeatures, uint64_t, NPY_UINT64)
#endif

#ifdef USE_FLOATMAX
	%template(LongRealFeatures) CDenseFeatures<floatmax_t>;
#endif

#ifdef USE_FLOAT32
	PROTOCOLS_DENSEFEATURES(ShortRealFeatures, float32_t, "f\0", NPY_FLOAT32)
	%template(ShortRealFeatures) CDenseFeatures<float32_t>;
	EXTEND_DENSEFEATURES(ShortRealFeatures, float32_t, NPY_FLOAT32)
#endif

#ifdef USE_FLOAT64
	PROTOCOLS_DENSEFEATURES(RealFeatures, float64_t, "d\0", NPY_FLOAT64)
	%template(RealFeatures) CDenseFeatures<float64_t>;
	EXTEND_DENSEFEATURES(RealFeatures, float64_t, NPY_FLOAT64)
#endif
}

/* Templated Class StreamingDenseFeatures */
%include <shogun/features/streaming/StreamingDenseFeatures.h>
namespace shogun
 {
#ifdef USE_BOOL
    %template(StreamingBoolFeatures) CStreamingDenseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingCharFeatures) CStreamingDenseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingByteFeatures) CStreamingDenseFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingWordFeatures) CStreamingDenseFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(StreamingShortFeatures) CStreamingDenseFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingIntFeatures)  CStreamingDenseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingUIntFeatures)  CStreamingDenseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingLongIntFeatures)  CStreamingDenseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingULongIntFeatures)  CStreamingDenseFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingLongRealFeatures) CStreamingDenseFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingShortRealFeatures) CStreamingDenseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingRealFeatures) CStreamingDenseFeatures<float64_t>;
#endif
}

/* these classes need the above typed CFeature definitions */
%rename(MeanShiftDataGenerator) CMeanShiftDataGenerator;
%include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

%rename(GaussianBlobsDataGenerator) CGaussianBlobsDataGenerator;
%include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>

/* Templated Class DenseSubsetFeatures */
%include <shogun/features/DenseSubsetFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(BoolSubsetFeatures) CDenseSubsetFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(CharSubsetFeatures) CDenseSubsetFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(ByteSubsetFeatures) CDenseSubsetFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordSubsetFeatures) CDenseSubsetFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortSubsetFeatures) CDenseSubsetFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntSubsetFeatures)  CDenseSubsetFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntSubsetFeatures)  CDenseSubsetFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntSubsetFeatures)  CDenseSubsetFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntSubsetFeatures)  CDenseSubsetFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSubsetFeatures) CDenseSubsetFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSubsetFeatures) CDenseSubsetFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealSubsetFeatures) CDenseSubsetFeatures<float64_t>;
#endif
}

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/IndexFeatures.h>
%include <shogun/features/AttributeFeatures.h>
%include <shogun/features/CombinedFeatures.h>
%include <shogun/features/CombinedDotFeatures.h>
%include <shogun/features/HashedDocDotFeatures.h>
%include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
%include <shogun/features/RandomKitchenSinksDotFeatures.h>
%include <shogun/features/RandomFourierDotFeatures.h>

%include <shogun/labels/Labels.h>
%include <shogun/labels/LabelsFactory.h>
%include <shogun/labels/DenseLabels.h>
%include <shogun/labels/BinaryLabels.h>
%include <shogun/labels/LatentLabels.h>
%include <shogun/labels/MulticlassLabels.h>
%include <shogun/labels/RegressionLabels.h>
%include <shogun/labels/StructuredLabels.h>
%include <shogun/labels/MultilabelLabels.h>

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
%include <shogun/features/LatentFeatures.h>
%include <shogun/features/MatrixFeatures.h>

/* Templated Class MatrixFeatures */
%include <shogun/features/MatrixFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(BoolMatrixFeatures) CMatrixFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(CharMatrixFeatures) CMatrixFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(ByteMatrixFeatures) CMatrixFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordMatrixFeatures) CMatrixFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortMatrixFeatures) CMatrixFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntMatrixFeatures)  CMatrixFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntMatrixFeatures)  CMatrixFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntMatrixFeatures)  CMatrixFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntMatrixFeatures)  CMatrixFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealMatrixFeatures) CMatrixFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealMatrixFeatures) CMatrixFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealMatrixFeatures) CMatrixFeatures<float64_t>;
#endif
}
