/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2009 Soeren Sonnenburg
 * Copyright (C) 2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

/* Remove C Prefix */
%rename(IOBuffer) CIOBuffer;

/* Templated Class ParseBuffer */
%include <shogun/io/ParseBuffer.h>
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
%include <shogun/io/InputParser.h>
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

%rename(File) CFile;
%rename(StreamingFile) CStreamingFile;
%rename(AsciiFile) CAsciiFile;
%rename(StreamingAsciiFile) CStreamingAsciiFile;
%rename(StreamingFileFromFeatures) CStreamingFileFromFeatures;
%rename(BinaryFile) CBinaryFile;
%rename(HDF5File) CHDF5File;
%rename(SerializableFile) CSerializableFile;
%rename(SerializableAsciiFile) CSerializableAsciiFile;
%rename(SerializableHdf5File) CSerializableHdf5File;
%rename(SerializableJsonFile) CSerializableJsonFile;
%rename(SerializableXmlFile) CSerializableXmlFile;
%rename(SimpleFile) CSimpleFile;
%rename(MemoryMappedFile) CMemoryMappedFile;

%include <shogun/io/File.h>
%include <shogun/io/StreamingFile.h>
%include <shogun/io/StreamingFileFromFeatures.h>

/* Template Class StreamingFileFromSparseFeatures */
%include <shogun/io/StreamingFileFromSparseFeatures.h>
namespace shogun
{
    %template(StreamingFileFromSparseBoolFeatures) CStreamingFileFromSparseFeatures<bool>;
    %template(StreamingFileFromSparseCharFeatures) CStreamingFileFromSparseFeatures<char>;
    %template(StreamingFileFromSparseByteFeatures) CStreamingFileFromSparseFeatures<uint8_t>;
    %template(StreamingFileFromSparseShortFeatures) CStreamingFileFromSparseFeatures<int16_t>;
    %template(StreamingFileFromSparseWordFeatures) CStreamingFileFromSparseFeatures<uint16_t>;
    %template(StreamingFileFromSparseIntFeatures) CStreamingFileFromSparseFeatures<int32_t>;
    %template(StreamingFileFromSparseUIntFeatures) CStreamingFileFromSparseFeatures<uint32_t>;
    %template(StreamingFileFromSparseLongFeatures) CStreamingFileFromSparseFeatures<int64_t>;
    %template(StreamingFileFromSparseUlongFeatures) CStreamingFileFromSparseFeatures<uint64_t>;
    %template(StreamingFileFromSparseShortRealFeatures) CStreamingFileFromSparseFeatures<float32_t>;
    %template(StreamingFileFromSparseRealFeatures) CStreamingFileFromSparseFeatures<float64_t>;
    %template(StreamingFileFromSparseLongRealFeatures) CStreamingFileFromSparseFeatures<floatmax_t>;
}

/* Template Class StreamingFileFromSimpleFeatures */
%include <shogun/io/StreamingFileFromSimpleFeatures.h>
namespace shogun
{
    %template(StreamingFileFromBoolFeatures) CStreamingFileFromSimpleFeatures<bool>;
    %template(StreamingFileFromCharFeatures) CStreamingFileFromSimpleFeatures<char>;
    %template(StreamingFileFromByteFeatures) CStreamingFileFromSimpleFeatures<uint8_t>;
    %template(StreamingFileFromShortFeatures) CStreamingFileFromSimpleFeatures<int16_t>;
    %template(StreamingFileFromWordFeatures) CStreamingFileFromSimpleFeatures<uint16_t>;
    %template(StreamingFileFromIntFeatures) CStreamingFileFromSimpleFeatures<int32_t>;
    %template(StreamingFileFromUIntFeatures) CStreamingFileFromSimpleFeatures<uint32_t>;
    %template(StreamingFileFromLongFeatures) CStreamingFileFromSimpleFeatures<int64_t>;
    %template(StreamingFileFromUlongFeatures) CStreamingFileFromSimpleFeatures<uint64_t>;
    %template(StreamingFileFromShortRealFeatures) CStreamingFileFromSimpleFeatures<float32_t>;
    %template(StreamingFileFromRealFeatures) CStreamingFileFromSimpleFeatures<float64_t>;
    %template(StreamingFileFromLongRealFeatures) CStreamingFileFromSimpleFeatures<floatmax_t>;
}


%include <shogun/io/AsciiFile.h>
%include <shogun/io/StreamingAsciiFile.h>
%include <shogun/io/BinaryFile.h>
%include <shogun/io/HDF5File.h>
%include <shogun/io/SerializableFile.h>
%include <shogun/io/SerializableAsciiFile.h>
%include <shogun/io/SerializableHdf5File.h>
%include <shogun/io/SerializableJsonFile.h>
%include <shogun/io/SerializableXmlFile.h>

%include <shogun/io/SimpleFile.h>
%include <shogun/io/MemoryMappedFile.h>
