/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saloni Nigam, Sergey Lisitsyn
 */

#ifdef HAVE_PYTHON
%feature("autodoc", "get_str(self) -> numpy 1dim array of str\n\nUse this instead of get_string() which is not nicely wrapped") get_str;
%feature("autodoc", "get_hist(self) -> numpy 1dim array of int") get_hist;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of int") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_fm(self) -> numpy 1dim array of float") get_fm;
%feature("autodoc", "get_labels(self) -> numpy 1dim array of float") get_labels;
#endif

#if defined(USE_SWIG_DIRECTORS) && defined(SWIGPYTHON)
%feature("director") shogun::DirectorDotFeatures;
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

%shared_ptr(shogun::Alphabet);
%shared_ptr(shogun::Features);
%shared_ptr(shogun::AttributeFeatures);
%shared_ptr(shogun::StreamingFeatures);
%shared_ptr(shogun::DotFeatures);
SHARED_RANDOM_INTERFACE(shogun::DotFeatures)
%shared_ptr(shogun::DirectorDotFeatures)
%shared_ptr(shogun::BinnedDotFeatures)
%shared_ptr(shogun::StreamingDotFeatures)

%shared_ptr(shogun::DummyFeatures)
%shared_ptr(shogun::IndexFeatures)
%shared_ptr(shogun::AttributeFeatures)
%shared_ptr(shogun::CombinedFeatures)
%shared_ptr(shogun::CombinedDotFeatures)
%shared_ptr(shogun::HashedDocDotFeatures)
%shared_ptr(shogun::StreamingHashedDocDotFeatures)
%shared_ptr(shogun::RandomKitchenSinksDotFeatures)
SHARED_RANDOM_INTERFACE(shogun::RandomKitchenSinksDotFeatures)
%shared_ptr(shogun::RandomFourierDotFeatures)
%shared_ptr(shogun::Labels)

PROTOCOLS_DENSELABELS(DenseLabels, DenseLabels, float64_t, "d\0", NPY_FLOAT64)
%shared_ptr(shogun::DenseLabels)

PROTOCOLS_DENSELABELS(BinaryLabels, BinaryLabels, float64_t, "d\0", NPY_FLOAT64)
%shared_ptr(shogun::BinaryLabels)

PROTOCOLS_DENSELABELS(MulticlassLabels, MulticlassLabels, float64_t, "d\0", NPY_FLOAT64)
%shared_ptr(shogun::MulticlassLabels)

PROTOCOLS_DENSELABELS(RegressionLabels, RegressionLabels, float64_t, "d\0", NPY_FLOAT64)
%shared_ptr(shogun::RegressionLabels)

%shared_ptr(shogun::StructuredLabels)
%shared_ptr(shogun::LatentLabels)
%shared_ptr(shogun::MultilabelLabels)
%shared_ptr(shogun::RealFileFeatures)
%shared_ptr(shogun::FKFeatures)
%shared_ptr(shogun::TOPFeatures)
%shared_ptr(shogun::SNPFeatures)
%shared_ptr(shogun::WDFeatures)
%shared_ptr(shogun::HashedWDFeatures)
%shared_ptr(shogun::HashedWDFeaturesTransposed)
%shared_ptr(shogun::PolyFeatures)
%shared_ptr(shogun::SparsePolyFeatures)
%shared_ptr(shogun::LBPPyrDotFeatures)
%shared_ptr(shogun::ExplicitSpecFeatures)
%shared_ptr(shogun::ImplicitWeightedSpecFeatures)
%shared_ptr(shogun::DataGenerator)
%shared_ptr(shogun::LatentFeatures)

#ifdef USE_BOOL
    %shared_ptr(shogun::StringFeatures<bool>)
    %shared_ptr(shogun::StreamingStringFeatures<bool>)
    %shared_ptr(shogun::StringFileFeatures<bool>)
    %shared_ptr(shogun::SparseFeatures<bool>)
    %shared_ptr(shogun::StreamingSparseFeatures<bool>)
    %shared_ptr(shogun::StreamingDenseFeatures<bool>)
    %shared_ptr(shogun::DenseSubsetFeatures<bool>)
    %shared_ptr(shogun::MatrixFeatures<bool>)
#endif
#ifdef USE_CHAR
    %shared_ptr(shogun::StringFeatures<char>)
    %shared_ptr(shogun::StreamingStringFeatures<char>)
    %shared_ptr(shogun::StringFileFeatures<char>)
    %shared_ptr(shogun::SparseFeatures<char>)
    %shared_ptr(shogun::StreamingSparseFeatures<char>)
    %shared_ptr(shogun::StreamingDenseFeatures<char>)
    %shared_ptr(shogun::DenseSubsetFeatures<char>)
    %shared_ptr(shogun::MatrixFeatures<char>)
#endif
#ifdef USE_UINT8
    %shared_ptr(shogun::StringFeatures<uint8_t>)
    %shared_ptr(shogun::StreamingStringFeatures<uint8_t>)
    %shared_ptr(shogun::StringFileFeatures<uint8_t>)
    %shared_ptr(shogun::SparseFeatures<uint8_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<uint8_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<uint8_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<uint8_t>)
    %shared_ptr(shogun::MatrixFeatures<uint8_t>)
#endif
#ifdef USE_INT16
    %shared_ptr(shogun::StringFeatures<int16_t>)
    %shared_ptr(shogun::StreamingStringFeatures<int16_t>)
    %shared_ptr(shogun::StringFileFeatures<int16_t>)
    %shared_ptr(shogun::SparseFeatures<int16_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<int16_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<int16_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<int16_t>)
    %shared_ptr(shogun::MatrixFeatures<int16_t>)
#endif
#ifdef USE_UINT16
    %shared_ptr(shogun::StringFeatures<uint16_t>)
    %shared_ptr(shogun::StreamingStringFeatures<uint16_t>)
    %shared_ptr(shogun::StringFileFeatures<uint16_t>)
    %shared_ptr(shogun::SparseFeatures<uint16_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<uint16_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<uint16_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<uint16_t>)
    %shared_ptr(shogun::MatrixFeatures<uint16_t>)
#endif
#ifdef USE_INT32
    %shared_ptr(shogun::StringFeatures<int32_t>)
    %shared_ptr(shogun::StreamingStringFeatures<int32_t>)
    %shared_ptr(shogun::StringFileFeatures<int32_t>)
    %shared_ptr(shogun::SparseFeatures<int32_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<int32_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<int32_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<int32_t>)
    %shared_ptr(shogun::MatrixFeatures<int32_t>)
#endif
#ifdef USE_UINT32
    %shared_ptr(shogun::StringFeatures<uint32_t>)
    %shared_ptr(shogun::StreamingStringFeatures<uint32_t>)
    %shared_ptr(shogun::StringFileFeatures<uint32_t>)
    %shared_ptr(shogun::SparseFeatures<uint32_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<uint32_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<uint32_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<uint32_t>)
    %shared_ptr(shogun::MatrixFeatures<uint32_t>)
#endif
#ifdef USE_INT64
    %shared_ptr(shogun::StringFeatures<int64_t>)
    %shared_ptr(shogun::StreamingStringFeatures<int64_t>)
    %shared_ptr(shogun::StringFileFeatures<int64_t>)
    %shared_ptr(shogun::SparseFeatures<int64_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<int64_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<int64_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<int64_t>)
    %shared_ptr(shogun::MatrixFeatures<int64_t>)
#endif
#ifdef USE_UINT64
    %shared_ptr(shogun::StringFeatures<uint64_t>)
    %shared_ptr(shogun::StreamingStringFeatures<uint64_t>)
    %shared_ptr(shogun::StringFileFeatures<uint64_t>)
    %shared_ptr(shogun::SparseFeatures<uint64_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<uint64_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<uint64_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<uint64_t>)
    %shared_ptr(shogun::MatrixFeatures<uint64_t>)
#endif
#ifdef USE_FLOAT32
    %shared_ptr(shogun::StringFeatures<float32_t>)
    %shared_ptr(shogun::StreamingStringFeatures<float32_t>)
    %shared_ptr(shogun::StringFileFeatures<float32_t>)
    %shared_ptr(shogun::SparseFeatures<float32_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<float32_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<float32_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<float32_t>)
    %shared_ptr(shogun::MatrixFeatures<float32_t>)
#endif
#ifdef USE_FLOAT64
    %shared_ptr(shogun::StringFeatures<float64_t>)
    %shared_ptr(shogun::StreamingStringFeatures<float64_t>)
    %shared_ptr(shogun::StringFileFeatures<float64_t>)
    %shared_ptr(shogun::SparseFeatures<float64_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<float64_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<float64_t>)
    %shared_ptr(shogun::DenseFeatures<float64_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<float64_t>)
    %shared_ptr(shogun::MatrixFeatures<float64_t>)
    %shared_ptr(shogun::DenseFeatures<float64_t>);
    %shared_ptr(shogun::Seedable<shogun::StreamingDenseFeatures<float64_t>>);
    %shared_ptr(shogun::RandomMixin<shogun::StreamingDenseFeatures<float64_t>, std::mt19937_64>)
#endif
#ifdef USE_FLOATMAX
    %shared_ptr(shogun::StringFeatures<floatmax_t>)
    %shared_ptr(shogun::StreamingStringFeatures<floatmax_t>)
    %shared_ptr(shogun::StringFileFeatures<floatmax_t>)
    %shared_ptr(shogun::SparseFeatures<floatmax_t>)
    %shared_ptr(shogun::StreamingSparseFeatures<floatmax_t>)
    %shared_ptr(shogun::StreamingDenseFeatures<floatmax_t>)
    %shared_ptr(shogun::DenseSubsetFeatures<floatmax_t>)
    %shared_ptr(shogun::MatrixFeatures<floatmax_t>)
#endif

/* Include Class Headers to make them visible from within the target language */
%include <shogun/features/FeatureTypes.h>
%include <shogun/features/Alphabet.h>
%include <shogun/features/Features.h>
%include <shogun/features/DotFeatures.h>
RANDOM_INTERFACE(DotFeatures)
%include <shogun/features/DirectorDotFeatures.h>
%include <shogun/features/BinnedDotFeatures.h>
%include <shogun/features/streaming/StreamingFeatures.h>
%include <shogun/features/streaming/StreamingDotFeatures.h>

%include <shogun/features/DataGenerator.h>

/* FIXME: DenseFeatures are still included/defined since certain classes
 * inherit from them (e.g. TOPKernel : DenseFeatures<float64_t>), as otherwise
 * SWIG will not treat those as subclasses of Features. Once all those classes
 * are instantiated w factories, this hack can be removed. In order prevent use
 * of RealFeatures etc, they are named differently (it only needs to be visible,
 * but is not used)
 */
%include <shogun/features/DenseFeatures.h>
namespace shogun
{
#ifdef USE_FLOAT64
	%template(RealFeaturesDEPRECATED) DenseFeatures<float64_t>;
#endif
}

/* Templated Class StringFeatures */
%include <shogun/features/StringFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StringBoolFeatures) StringFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StringCharFeatures) StringFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StringByteFeatures) StringFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StringShortFeatures) StringFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StringWordFeatures) StringFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringIntFeatures) StringFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StringUIntFeatures) StringFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StringLongFeatures) StringFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StringUlongFeatures) StringFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StringShortRealFeatures) StringFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringRealFeatures) StringFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StringLongRealFeatures) StringFeatures<floatmax_t>;
#endif
}

/* Templated Class StreamingStringFeatures */
%include <shogun/features/streaming/StreamingStringFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingStringBoolFeatures) StreamingStringFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingStringCharFeatures) StreamingStringFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingStringByteFeatures) StreamingStringFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingStringShortFeatures) StreamingStringFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingStringWordFeatures) StreamingStringFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingStringIntFeatures) StreamingStringFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingStringUIntFeatures) StreamingStringFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingStringLongFeatures) StreamingStringFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingStringUlongFeatures) StreamingStringFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingStringShortRealFeatures) StreamingStringFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingStringRealFeatures) StreamingStringFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingStringLongRealFeatures) StreamingStringFeatures<floatmax_t>;
#endif
}

/* Templated Class StringFileFeatures */
%include <shogun/features/StringFileFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StringFileBoolFeatures) StringFileFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StringFileCharFeatures) StringFileFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StringFileByteFeatures) StringFileFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StringFileShortFeatures) StringFileFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StringFileWordFeatures) StringFileFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StringFileIntFeatures) StringFileFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StringFileUIntFeatures) StringFileFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StringFileLongFeatures) StringFileFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StringFileUlongFeatures) StringFileFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StringFileShortRealFeatures) StringFileFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StringFileRealFeatures) StringFileFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StringFileLongRealFeatures) StringFileFeatures<floatmax_t>;
#endif
}

/* Templated Class SparseFeatures */
%include <shogun/features/SparseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(SparseBoolFeatures) SparseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(SparseCharFeatures) SparseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(SparseByteFeatures) SparseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(SparseShortFeatures) SparseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(SparseWordFeatures) SparseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(SparseIntFeatures) SparseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(SparseUIntFeatures) SparseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(SparseLongFeatures) SparseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(SparseUlongFeatures) SparseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(SparseShortRealFeatures) SparseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(SparseRealFeatures) SparseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(SparseLongRealFeatures) SparseFeatures<floatmax_t>;
#endif
}

/* Templated Class StreamingSparseFeatures */
%include <shogun/features/streaming/StreamingSparseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingSparseBoolFeatures) StreamingSparseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingSparseCharFeatures) StreamingSparseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingSparseByteFeatures) StreamingSparseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingSparseShortFeatures) StreamingSparseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingSparseWordFeatures) StreamingSparseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingSparseIntFeatures) StreamingSparseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingSparseUIntFeatures) StreamingSparseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingSparseLongFeatures) StreamingSparseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingSparseUlongFeatures) StreamingSparseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingSparseShortRealFeatures) StreamingSparseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingSparseRealFeatures) StreamingSparseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingSparseLongRealFeatures) StreamingSparseFeatures<floatmax_t>;
#endif
}

/* Templated Class StreamingDenseFeatures */
%include <shogun/features/streaming/StreamingDenseFeatures.h>
namespace shogun
 {
#ifdef USE_BOOL
    %template(StreamingBoolFeatures) StreamingDenseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingCharFeatures) StreamingDenseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingByteFeatures) StreamingDenseFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingWordFeatures) StreamingDenseFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(StreamingShortFeatures) StreamingDenseFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingIntFeatures)  StreamingDenseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingUIntFeatures)  StreamingDenseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingLongIntFeatures)  StreamingDenseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingULongIntFeatures)  StreamingDenseFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingLongRealFeatures) StreamingDenseFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingShortRealFeatures) StreamingDenseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingRealFeatures) StreamingDenseFeatures<float64_t>;

    /** Instantiate RandomMixin */
    %template(SeedableStreamingDense) shogun::Seedable<shogun::StreamingDenseFeatures<float64_t>>;
    %template(RandomMixinStreamingDense) shogun::RandomMixin<shogun::StreamingDenseFeatures<float64_t>, std::mt19937_64>;
#endif
}

/* these classes need the above typed CFeature definitions */
%shared_ptr(shogun::MeanShiftDataGenerator)
%include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>

%shared_ptr(shogun::GaussianBlobsDataGenerator)
%include <shogun/features/streaming/generators/GaussianBlobsDataGenerator.h>

/* Templated Class DenseSubsetFeatures */
%include <shogun/features/DenseSubsetFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(BoolSubsetFeatures) DenseSubsetFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(CharSubsetFeatures) DenseSubsetFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(ByteSubsetFeatures) DenseSubsetFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordSubsetFeatures) DenseSubsetFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortSubsetFeatures) DenseSubsetFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntSubsetFeatures)  DenseSubsetFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntSubsetFeatures)  DenseSubsetFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntSubsetFeatures)  DenseSubsetFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntSubsetFeatures)  DenseSubsetFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealSubsetFeatures) DenseSubsetFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealSubsetFeatures) DenseSubsetFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealSubsetFeatures) DenseSubsetFeatures<float64_t>;
#endif
}

%include <shogun/features/DummyFeatures.h>
%include <shogun/features/IndexFeatures.h>
%include <shogun/features/AttributeFeatures.h>
%include <shogun/features/CombinedFeatures.h>
%include <shogun/features/CombinedDotFeatures.h>
%include <shogun/features/hashed/HashedDocDotFeatures.h>
%include <shogun/features/streaming/StreamingHashedDocDotFeatures.h>
%include <shogun/features/RandomKitchenSinksDotFeatures.h>
RANDOM_INTERFACE(RandomKitchenSinksDotFeatures)
%include <shogun/features/RandomFourierDotFeatures.h>

%include <shogun/labels/Labels.h>
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
%include <shogun/features/hashed/HashedWDFeatures.h>
%include <shogun/features/hashed/HashedWDFeaturesTransposed.h>
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
    %template(BoolMatrixFeatures) MatrixFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(CharMatrixFeatures) MatrixFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(ByteMatrixFeatures) MatrixFeatures<uint8_t>;
#endif
#ifdef USE_UINT16
    %template(WordMatrixFeatures) MatrixFeatures<uint16_t>;
#endif
#ifdef USE_INT16
    %template(ShortMatrixFeatures) MatrixFeatures<int16_t>;
#endif
#ifdef USE_INT32
    %template(IntMatrixFeatures)  MatrixFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(UIntMatrixFeatures)  MatrixFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(LongIntMatrixFeatures)  MatrixFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(ULongIntMatrixFeatures)  MatrixFeatures<uint64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(LongRealMatrixFeatures) MatrixFeatures<floatmax_t>;
#endif
#ifdef USE_FLOAT32
    %template(ShortRealMatrixFeatures) MatrixFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(RealMatrixFeatures) MatrixFeatures<float64_t>;
#endif
}
