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
%include "Library_includes.i"
%include "Streaming_includes.i"

%import "Features.i"

/* Remove C Prefix */
%rename(IOBuffer) CIOBuffer;
%rename(StreamingFile) CStreamingFile;
%rename(StreamingAsciiFile) CStreamingAsciiFile;
%rename(StreamingFileFromFeatures) CStreamingFileFromFeatures;

/* Templated Class ParseBuffer */
%include <shogun/lib/ParseBuffer.h>
namespace shogun
{
    %template(ParseBoolBuffer) CParseBuffer<bool>;
    %template(ParseCharBuffer) CParseBuffer<char>;
    %template(ParseByteBuffer) CParseBuffer<uint8_t>;
    %template(ParseShortBuffer) CParseBuffer<int16_t>;
    %template(ParseWordBuffer) CParseBuffer<uint16_t>;
    %template(ParseIntBuffer) CParseBuffer<int32_t>;
    %template(ParseUIntBuffer) CParseBuffer<uint32_t>;
    %template(ParseLongBuffer) CParseBuffer<int64_t>;
    %template(ParseUlongBuffer) CParseBuffer<uint64_t>;
    %template(ParseShortRealBuffer) CParseBuffer<float32_t>;
    %template(ParseRealBuffer) CParseBuffer<float64_t>;
    %template(ParseLongRealBuffer) CParseBuffer<floatmax_t>;

    %template(ParseSparseBoolBuffer) CParseBuffer< SGSparseVectorEntry<bool> >;
    %template(ParseSparseCharBuffer) CParseBuffer< SGSparseVectorEntry<char> >;
    %template(ParseSparseByteBuffer) CParseBuffer< SGSparseVectorEntry<uint8_t> >;
    %template(ParseSparseShortBuffer) CParseBuffer< SGSparseVectorEntry<int16_t> >;
    %template(ParseSparseWordBuffer) CParseBuffer< SGSparseVectorEntry<uint16_t> >;
    %template(ParseSparseIntBuffer) CParseBuffer< SGSparseVectorEntry<int32_t> >;
    %template(ParseSparseUIntBuffer) CParseBuffer< SGSparseVectorEntry<uint32_t> >;
    %template(ParseSparseLongBuffer) CParseBuffer< SGSparseVectorEntry<int64_t> >;
    %template(ParseSparseUlongBuffer) CParseBuffer< SGSparseVectorEntry<uint64_t> >;
    %template(ParseSparseShortRealBuffer) CParseBuffer< SGSparseVectorEntry<float32_t> >;
    %template(ParseSparseRealBuffer) CParseBuffer< SGSparseVectorEntry<float64_t> >;
    %template(ParseSparseLongRealBuffer) CParseBuffer< SGSparseVectorEntry<floatmax_t> >;
}

/* Templated Class InputParser */
%include <shogun/lib/InputParser.h>
namespace shogun
{
    %template(InputBoolParser) CInputParser<bool>;
    %template(InputCharParser) CInputParser<char>;
    %template(InputByteParser) CInputParser<uint8_t>;
    %template(InputShortParser) CInputParser<int16_t>;
    %template(InputWordParser) CInputParser<uint16_t>;
    %template(InputIntParser) CInputParser<int32_t>;
    %template(InputUIntParser) CInputParser<uint32_t>;
    %template(InputLongParser) CInputParser<int64_t>;
    %template(InputUlongParser) CInputParser<uint64_t>;
    %template(InputShortRealParser) CInputParser<float32_t>;
    %template(InputRealParser) CInputParser<float64_t>;
    %template(InputLongRealParser) CInputParser<floatmax_t>;

    %template(InputSparseBoolParser) CInputParser< SGSparseVectorEntry<bool> >;
    %template(InputSparseCharParser) CInputParser< SGSparseVectorEntry<char> >;
    %template(InputSparseByteParser) CInputParser< SGSparseVectorEntry<uint8_t> >;
    %template(InputSparseShortParser) CInputParser< SGSparseVectorEntry<int16_t> >;
    %template(InputSparseWordParser) CInputParser< SGSparseVectorEntry<uint16_t> >;
    %template(InputSparseIntParser) CInputParser< SGSparseVectorEntry<int32_t> >;
    %template(InputSparseUIntParser) CInputParser< SGSparseVectorEntry<uint32_t> >;
    %template(InputSparseLongParser) CInputParser< SGSparseVectorEntry<int64_t> >;
    %template(InputSparseUlongParser) CInputParser< SGSparseVectorEntry<uint64_t> >;
    %template(InputSparseShortRealParser) CInputParser< SGSparseVectorEntry<float32_t> >;
    %template(InputSparseRealParser) CInputParser< SGSparseVectorEntry<float64_t> >;
    %template(InputSparseLongRealParser) CInputParser< SGSparseVectorEntry<floatmax_t> >;
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

/* Templated Class StreamingFileFromStringFeatures */
%include <shogun/lib/StreamingFileFromStringFeatures.h>
namespace shogun
{
    %template(StreamingFileFromStringFeaturesBool) CStreamingFileFromStringFeatures<bool>;
    %template(StreamingFileFromStringFeaturesChar) CStreamingFileFromStringFeatures<char>;
    %template(StreamingFileFromStringFeaturesByte) CStreamingFileFromStringFeatures<uint8_t>;
    %template(StreamingFileFromStringFeaturesShort) CStreamingFileFromStringFeatures<int16_t>;
    %template(StreamingFileFromStringFeaturesWord) CStreamingFileFromStringFeatures<uint16_t>;
    %template(StreamingFileFromStringFeaturesInt) CStreamingFileFromStringFeatures<int32_t>;
    %template(StreamingFileFromStringFeaturesUInt) CStreamingFileFromStringFeatures<uint32_t>;
    %template(StreamingFileFromStringFeaturesLong) CStreamingFileFromStringFeatures<int64_t>;
    %template(StreamingFileFromStringFeaturesUlong) CStreamingFileFromStringFeatures<uint64_t>;
    %template(StreamingFileFromStringFeaturesShortReal) CStreamingFileFromStringFeatures<float32_t>;
    %template(StreamingFileFromStringFeaturesReal) CStreamingFileFromStringFeatures<float64_t>;
    %template(StreamingFileFromStringFeaturesLongReal) CStreamingFileFromStringFeatures<floatmax_t>;
}

/* StreamingFeatures */

/* Remove C Prefix */
%rename(StreamingFeatures) CStreamingFeatures;
%rename(StreamingDotFeatures) CStreamingDotFeatures;

/* Templated Class StreamingSimpleFeatures */
%include <shogun/lib/StreamingSimpleFeatures.h>
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
%include <shogun/lib/StreamingSparseFeatures.h>
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

/* Templated Class StreamingStringFeatures */
%include <shogun/lib/StreamingStringFeatures.h>
namespace shogun
{
    %template(StreamingStringFeaturesBool) CStreamingStringFeatures<bool>;
    %template(StreamingStringFeaturesChar) CStreamingStringFeatures<char>;
    %template(StreamingStringFeaturesByte) CStreamingStringFeatures<uint8_t>;
    %template(StreamingStringFeaturesShort) CStreamingStringFeatures<int16_t>;
    %template(StreamingStringFeaturesWord) CStreamingStringFeatures<uint16_t>;
    %template(StreamingStringFeaturesInt) CStreamingStringFeatures<int32_t>;
    %template(StreamingStringFeaturesUInt) CStreamingStringFeatures<uint32_t>;
    %template(StreamingStringFeaturesLong) CStreamingStringFeatures<int64_t>;
    %template(StreamingStringFeaturesUlong) CStreamingStringFeatures<uint64_t>;
    %template(StreamingStringFeaturesShortReal) CStreamingStringFeatures<float32_t>;
    %template(StreamingStringFeaturesReal) CStreamingStringFeatures<float64_t>;
    %template(StreamingStringFeaturesLongReal) CStreamingStringFeatures<floatmax_t>;
}
