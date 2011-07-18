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
"The `Library` module gathers all miscellaneous Objects in the SHOGUN toolkit."
%enddef

%module(docstring=DOCSTR) Library
#undef DOCSTR

/* Documentation */
%feature("autodoc","0");

#ifdef HAVE_DOXYGEN
#ifndef SWIGRUBY
%include "Library_doxygen.i"
#endif
#endif

/* Include Module Definitions */
%include "SGBase.i"
%include "Library_includes.i"
%include "Features_includes.i"

/* Remove C Prefix */
%rename(IOBuffer) CIOBuffer;

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

%rename(Cache) CCache;
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
%rename(ListElement) CListElement;
%rename(List) CList;
%rename(Math) CMath;
%rename(Signal) CSignal;
%rename(SimpleFile) CSimpleFile;
%rename(Time) CTime;
%rename(Hash) CHash;
%rename(MemoryMappedFile) CMemoryMappedFile;

%ignore RADIX_STACK_SIZE;
%ignore NUMTRAPPEDSIGS;
%ignore TRIE_TERMINAL_CHARACTER;
%ignore NO_CHILD;

#pragma SWIG nowarn=312,362,389
%warnfilter(509) CArray;
%warnfilter(509) CArray2;
%warnfilter(509) CArray3;

/* Include Class Headers to make them visible from within the target language */
/* Template Class DynamicArray */
%include <shogun/lib/DynamicArray.h>
%include <shogun/base/DynArray.h>

namespace shogun
{
    %template(DynamicCharArray) CDynamicArray<char>;
    %template(DynamicByteArray) CDynamicArray<uint8_t>;
    %template(DynamicShortArray) CDynamicArray<int16_t>;
    %template(DynamicWordArray) CDynamicArray<uint16_t>;
    %template(DynamicIntArray) CDynamicArray<int32_t>;
    %template(DynamicUIntArray) CDynamicArray<uint32_t>;
    %template(DynamicLongArray) CDynamicArray<int64_t>;
    %template(DynamicULongArray) CDynamicArray<uint64_t>;
    %template(DynamicShortRealArray) CDynamicArray<float32_t>;
    %template(DynamicRealArray) CDynamicArray<float64_t>;
    %template(DynamicPlifArray) DynArray<shogun::CPlifBase*>;
}

/* Template Class GCArray */
%include <shogun/lib/GCArray.h>
namespace shogun
{
    %template(PlifGCArray) CGCArray<shogun::CPlifBase*>;
}

/* Hash */
%include <shogun/lib/Hash.h>

/* Template Class Array */
%include <shogun/lib/Array.h>
namespace shogun
{
    %template(CharArray) CArray<char>;
    %template(ByteArray) CArray<uint8_t>;
    %template(ShortArray) CArray<int16_t>;
    %template(WordArray) CArray<uint16_t>;
    %template(IntArray) CArray<int32_t>;
    %template(UIntArray) CArray<uint32_t>;
    %template(LongArray) CArray<int64_t>;
    %template(ULongArray) CArray<uint64_t>;
    %template(ShortRealArray) CArray<float32_t>;
    %template(RealArray) CArray<float64_t>;
}

/* Template Class Array2 */
%include <shogun/lib/Array2.h>
namespace shogun
{
    %template(CharArray2) CArray2<char>;
    %template(ByteArray2) CArray2<uint8_t>;
    %template(ShortArray2) CArray2<int16_t>;
    %template(WordArray2) CArray2<uint16_t>;
    %template(IntArray2) CArray2<int32_t>;
    %template(UIntArray2) CArray2<uint32_t>;
    %template(LongArray2) CArray2<int64_t>;
    %template(ULongArray2) CArray2<uint64_t>;
    %template(ShortRealArray2) CArray2<float32_t>;
    %template(RealArray2) CArray2<float64_t>;
}

/* Template Class Array3 */
%include <shogun/lib/Array3.h>
namespace shogun
{
    %template(CharArray3) CArray3<char>;
    %template(ByteArray3) CArray3<uint8_t>;
    %template(ShortArray3) CArray3<int16_t>;
    %template(WordArray3) CArray3<uint16_t>;
    %template(IntArray3) CArray3<int32_t>;
    %template(UIntArray3) CArray3<uint32_t>;
    %template(LongArray3) CArray3<int64_t>;
    %template(ULongArray3) CArray3<uint64_t>;
    %template(ShortRealArray3) CArray3<float32_t>;
    %template(RealArray3) CArray3<float64_t>;
}

%include <shogun/lib/Cache.h>
%include <shogun/lib/GCArray.h>
%include <shogun/lib/File.h>
%include <shogun/lib/StreamingFile.h>
%include <shogun/lib/StreamingFileFromFeatures.h>

/* Template Class StreamingFileFromSparseFeatures */
%include <shogun/lib/StreamingFileFromSparseFeatures.h>
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
%include <shogun/lib/StreamingFileFromSimpleFeatures.h>
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


%include <shogun/lib/AsciiFile.h>
%include <shogun/lib/StreamingAsciiFile.h>
%include <shogun/lib/BinaryFile.h>
%include <shogun/lib/HDF5File.h>
%include <shogun/lib/SerializableFile.h>
%include <shogun/lib/SerializableAsciiFile.h>
%include <shogun/lib/SerializableHdf5File.h>
%include <shogun/lib/SerializableJsonFile.h>
%include <shogun/lib/SerializableXmlFile.h>

%include <shogun/lib/List.h>
%include <shogun/lib/Mathematics.h>
%include <shogun/lib/Signal.h>
%include <shogun/lib/SimpleFile.h>
%include <shogun/lib/Time.h>
%include <shogun/lib/Trie.h>
%include <shogun/lib/MemoryMappedFile.h>
%include <shogun/lib/Compressor.h>
