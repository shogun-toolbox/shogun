%define STREAMING_DOCSTR
"The `Streaming` module gathers all base methods for StreamingFeatures in Shogun."
%enddef

%module(docstring=STREAMING_DOCSTR) Streaming
#undef DOCSTR

 /* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Streaming_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Features_includes.i"
%include "Preprocessor_includes.i"
%include "Distribution_includes.i"
%include "Library_includes.i"
%include "Kernel_includes.i"
%include "Distance_includes.i"
%include "Streaming_includes.i"

%import "Preprocessor.i"
%import "Distribution.i"
%import "Library.i"
%import "Features.i"

/* Remove C Prefix */
%rename(IOBuffer) CIOBuffer;
%rename(StreamingFeatures) CStreamingFeatures;
%rename(StreamingDotFeatures) CStreamingDotFeatures;
%rename(StreamingFile) CStreamingFile;
%rename(StreamingAsciiFile) CStreamingAsciiFile;
%rename(StreamingFileFromFeatures) CStreamingFileFromFeatures;

/* Templated Class ParseBuffer */
%include <shogun/lib/ParseBuffer.h>
namespace shogun
{
    %template(ExamplesRingBool) CParseBuffer<bool>;
    %template(ExamplesRingChar) CParseBuffer<char>;
    %template(ExamplesRingByte) CParseBuffer<uint8_t>;
    %template(ExamplesRingShort) CParseBuffer<int16_t>;
    %template(ExamplesRingWord) CParseBuffer<uint16_t>;
    %template(ExamplesRingInt) CParseBuffer<int32_t>;
    %template(ExamplesRingUInt) CParseBuffer<uint32_t>;
    %template(ExamplesRingLong) CParseBuffer<int64_t>;
    %template(ExamplesRingUlong) CParseBuffer<uint64_t>;
    %template(ExamplesRingShortReal) CParseBuffer<float32_t>;
    %template(ExamplesRingReal) CParseBuffer<float64_t>;
    %template(ExamplesRingLongReal) CParseBuffer<floatmax_t>;

    %template(ExamplesRingSparseBool) CParseBuffer< SGSparseVectorEntry<bool> >;
    %template(ExamplesRingSparseChar) CParseBuffer< SGSparseVectorEntry<char> >;
    %template(ExamplesRingSparseByte) CParseBuffer< SGSparseVectorEntry<uint8_t> >;
    %template(ExamplesRingSparseShort) CParseBuffer< SGSparseVectorEntry<int16_t> >;
    %template(ExamplesRingSparseWord) CParseBuffer< SGSparseVectorEntry<uint16_t> >;
    %template(ExamplesRingSparseInt) CParseBuffer< SGSparseVectorEntry<int32_t> >;
    %template(ExamplesRingSparseUInt) CParseBuffer< SGSparseVectorEntry<uint32_t> >;
    %template(ExamplesRingSparseLong) CParseBuffer< SGSparseVectorEntry<int64_t> >;
    %template(ExamplesRingSparseUlong) CParseBuffer< SGSparseVectorEntry<uint64_t> >;
    %template(ExamplesRingSparseShortReal) CParseBuffer< SGSparseVectorEntry<float32_t> >;
    %template(ExamplesRingSparseReal) CParseBuffer< SGSparseVectorEntry<float64_t> >;
    %template(ExamplesRingSparseLongReal) CParseBuffer< SGSparseVectorEntry<floatmax_t> >;
}

/* Templated Class InputParser */
%include <shogun/lib/InputParser.h>
namespace shogun
{
    %template(ParserBool) CInputParser<bool>;
    %template(ParserChar) CInputParser<char>;
    %template(ParserByte) CInputParser<uint8_t>;
    %template(ParserShort) CInputParser<int16_t>;
    %template(ParserWord) CInputParser<uint16_t>;
    %template(ParserInt) CInputParser<int32_t>;
    %template(ParserUInt) CInputParser<uint32_t>;
    %template(ParserLong) CInputParser<int64_t>;
    %template(ParserUlong) CInputParser<uint64_t>;
    %template(ParserShortReal) CInputParser<float32_t>;
    %template(ParserReal) CInputParser<float64_t>;
    %template(ParserLongReal) CInputParser<floatmax_t>;

    %template(ParserSparseBool) CInputParser< SGSparseVectorEntry<bool> >;
    %template(ParserSparseChar) CInputParser< SGSparseVectorEntry<char> >;
    %template(ParserSparseByte) CInputParser< SGSparseVectorEntry<uint8_t> >;
    %template(ParserSparseShort) CInputParser< SGSparseVectorEntry<int16_t> >;
    %template(ParserSparseWord) CInputParser< SGSparseVectorEntry<uint16_t> >;
    %template(ParserSparseInt) CInputParser< SGSparseVectorEntry<int32_t> >;
    %template(ParserSparseUInt) CInputParser< SGSparseVectorEntry<uint32_t> >;
    %template(ParserSparseLong) CInputParser< SGSparseVectorEntry<int64_t> >;
    %template(ParserSparseUlong) CInputParser< SGSparseVectorEntry<uint64_t> >;
    %template(ParserSparseShortReal) CInputParser< SGSparseVectorEntry<float32_t> >;
    %template(ParserSparseReal) CInputParser< SGSparseVectorEntry<float64_t> >;
    %template(ParserSparseLongReal) CInputParser< SGSparseVectorEntry<floatmax_t> >;
}

/* Templated Class StreamingFileFromSimpleFeatures */
%include <shogun/lib/StreamingFileFromSimpleFeatures.h>
namespace shogun
{
    %template(StreamingFileFromSimpleFeaturesBool) CStreamingFileFromSimpleFeatures<bool>;
    %template(StreamingFileFromSimpleFeaturesChar) CStreamingFileFromSimpleFeatures<char>;
    %template(StreamingFileFromSimpleFeaturesByte) CStreamingFileFromSimpleFeatures<uint8_t>;
    %template(StreamingFileFromSimpleFeaturesShort) CStreamingFileFromSimpleFeatures<int16_t>;
    %template(StreamingFileFromSimpleFeaturesWord) CStreamingFileFromSimpleFeatures<uint16_t>;
    %template(StreamingFileFromSimpleFeaturesInt) CStreamingFileFromSimpleFeatures<int32_t>;
    %template(StreamingFileFromSimpleFeaturesUInt) CStreamingFileFromSimpleFeatures<uint32_t>;
    %template(StreamingFileFromSimpleFeaturesLong) CStreamingFileFromSimpleFeatures<int64_t>;
    %template(StreamingFileFromSimpleFeaturesUlong) CStreamingFileFromSimpleFeatures<uint64_t>;
    %template(StreamingFileFromSimpleFeaturesShortReal) CStreamingFileFromSimpleFeatures<float32_t>;
    %template(StreamingFileFromSimpleFeaturesReal) CStreamingFileFromSimpleFeatures<float64_t>;
    %template(StreamingFileFromSimpleFeaturesLongReal) CStreamingFileFromSimpleFeatures<floatmax_t>;
}

/* Templated Class StreamingFileFromSparseFeatures */
%include <shogun/lib/StreamingFileFromSparseFeatures.h>
namespace shogun
{
    %template(StreamingFileFromSparseFeaturesBool) CStreamingFileFromSparseFeatures<bool>;
    %template(StreamingFileFromSparseFeaturesChar) CStreamingFileFromSparseFeatures<char>;
    %template(StreamingFileFromSparseFeaturesByte) CStreamingFileFromSparseFeatures<uint8_t>;
    %template(StreamingFileFromSparseFeaturesShort) CStreamingFileFromSparseFeatures<int16_t>;
    %template(StreamingFileFromSparseFeaturesWord) CStreamingFileFromSparseFeatures<uint16_t>;
    %template(StreamingFileFromSparseFeaturesInt) CStreamingFileFromSparseFeatures<int32_t>;
    %template(StreamingFileFromSparseFeaturesUInt) CStreamingFileFromSparseFeatures<uint32_t>;
    %template(StreamingFileFromSparseFeaturesLong) CStreamingFileFromSparseFeatures<int64_t>;
    %template(StreamingFileFromSparseFeaturesUlong) CStreamingFileFromSparseFeatures<uint64_t>;
    %template(StreamingFileFromSparseFeaturesShortReal) CStreamingFileFromSparseFeatures<float32_t>;
    %template(StreamingFileFromSparseFeaturesReal) CStreamingFileFromSparseFeatures<float64_t>;
    %template(StreamingFileFromSparseFeaturesLongReal) CStreamingFileFromSparseFeatures<floatmax_t>;
}

/* Templated Class StreamingSimpleFeatures */
%include <shogun/features/StreamingSimpleFeatures.h>
namespace shogun
{
    %template(StreamingSimpleFeaturesBool) CStreamingSimpleFeatures<bool>;
    %template(StreamingSimpleFeaturesChar) CStreamingSimpleFeatures<char>;
    %template(StreamingSimpleFeaturesByte) CStreamingSimpleFeatures<uint8_t>;
    %template(StreamingSimpleFeaturesShort) CStreamingSimpleFeatures<int16_t>;
    %template(StreamingSimpleFeaturesWord) CStreamingSimpleFeatures<uint16_t>;
    %template(StreamingSimpleFeaturesInt) CStreamingSimpleFeatures<int32_t>;
    %template(StreamingSimpleFeaturesUInt) CStreamingSimpleFeatures<uint32_t>;
    %template(StreamingSimpleFeaturesLong) CStreamingSimpleFeatures<int64_t>;
    %template(StreamingSimpleFeaturesUlong) CStreamingSimpleFeatures<uint64_t>;
    %template(StreamingSimpleFeaturesShortReal) CStreamingSimpleFeatures<float32_t>;
    %template(StreamingSimpleFeaturesReal) CStreamingSimpleFeatures<float64_t>;
    %template(StreamingSimpleFeaturesLongReal) CStreamingSimpleFeatures<floatmax_t>;
}

/* Templated Class StreamingSparseFeatures */
%include <shogun/features/StreamingSparseFeatures.h>
namespace shogun
{
    %template(StreamingSparseFeaturesBool) CStreamingSparseFeatures<bool>;
    %template(StreamingSparseFeaturesChar) CStreamingSparseFeatures<char>;
    %template(StreamingSparseFeaturesByte) CStreamingSparseFeatures<uint8_t>;
    %template(StreamingSparseFeaturesShort) CStreamingSparseFeatures<int16_t>;
    %template(StreamingSparseFeaturesWord) CStreamingSparseFeatures<uint16_t>;
    %template(StreamingSparseFeaturesInt) CStreamingSparseFeatures<int32_t>;
    %template(StreamingSparseFeaturesUInt) CStreamingSparseFeatures<uint32_t>;
    %template(StreamingSparseFeaturesLong) CStreamingSparseFeatures<int64_t>;
    %template(StreamingSparseFeaturesUlong) CStreamingSparseFeatures<uint64_t>;
    %template(StreamingSparseFeaturesShortReal) CStreamingSparseFeatures<float32_t>;
    %template(StreamingSparseFeaturesReal) CStreamingSparseFeatures<float64_t>;
    %template(StreamingSparseFeaturesLongReal) CStreamingSparseFeatures<floatmax_t>;
}

%include <shogun/lib/IOBuffer.h>
%include <shogun/lib/StreamingFile.h>
%include <shogun/lib/StreamingAsciiFile.h>

%include <shogun/features/StreamingFeatures.h>
%include <shogun/features/StreamingDotFeatures.h>
%include <shogun/lib/StreamingFileFromFeatures.h>
