/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Saloni Nigam, Sergey Lisitsyn
 */

/* Remove C Prefix */
%shared_ptr(shogun::IOBuffer)

%shared_ptr(shogun::File)
%shared_ptr(shogun::StreamingFile)
%shared_ptr(shogun::CSVFile)
%shared_ptr(shogun::LibSVMFile)
%shared_ptr(shogun::StreamingAsciiFile)
%shared_ptr(shogun::io::InputStream)
%shared_ptr(shogun::io::OutputStream)
%shared_ptr(shogun::io::Serializer)
%shared_ptr(shogun::io::Deserializer)
%shared_ptr(shogun::io::BitserySerializer)
%shared_ptr(shogun::io::BitseryDeserializer)
%shared_ptr(shogun::io::JsonSerializer)
%shared_ptr(shogun::io::JsonDeserializer)
%shared_ptr(shogun::io::ByteArrayInputStream)
%shared_ptr(shogun::io::ByteArrayOutputStream)

%shared_ptr(shogun::StreamingFileFromFeatures)
%shared_ptr(shogun::BinaryFile)
%shared_ptr(shogun::HDF5File)
%shared_ptr(shogun::SimpleFile)
%shared_ptr(shogun::MemoryMappedFile)
%shared_ptr(shogun::Compressor)

#ifdef USE_BOOL
%shared_ptr(shogun::StreamingFileFromSparseFeatures<bool>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<bool>)
#endif
#ifdef USE_CHAR
%shared_ptr(shogun::StreamingFileFromSparseFeatures<char>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<char>)
#endif
#ifdef USE_UINT8
%shared_ptr(shogun::StreamingFileFromSparseFeatures<uint8_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<uint8_t>)
#endif
#ifdef USE_INT16
%shared_ptr(shogun::StreamingFileFromSparseFeatures<int16_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<int16_t>)
#endif
#ifdef USE_UINT16
%shared_ptr(shogun::StreamingFileFromSparseFeatures<uint16_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<uint16_t>)
#endif
#ifdef USE_INT32
%shared_ptr(shogun::StreamingFileFromSparseFeatures<int32_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<int32_t>)
#endif
#ifdef USE_UINT32
%shared_ptr(shogun::StreamingFileFromSparseFeatures<uint32_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<uint32_t>)
#endif
#ifdef USE_INT64
%shared_ptr(shogun::StreamingFileFromSparseFeatures<int64_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<int64_t>)
#endif
#ifdef USE_UINT64
%shared_ptr(shogun::StreamingFileFromSparseFeatures<uint64_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<uint64_t>)
#endif
#ifdef USE_FLOAT32
%shared_ptr(shogun::StreamingFileFromSparseFeatures<float32_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<float32_t>)
#endif
#ifdef USE_FLOAT64
%shared_ptr(shogun::StreamingFileFromSparseFeatures<float64_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<float64_t>)
#endif
#ifdef USE_FLOATMAX
%shared_ptr(shogun::StreamingFileFromSparseFeatures<floatmax_t>)
%shared_ptr(shogun::StreamingFileFromDenseFeatures<floatmax_t>)
#endif

%include <shogun/io/File.h>
%include <shogun/io/streaming/StreamingFile.h>
%include <shogun/io/streaming/StreamingFileFromFeatures.h>

/* Template Class StreamingFileFromSparseFeatures */
%include <shogun/io/streaming/StreamingFileFromSparseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingFileFromSparseBoolFeatures) StreamingFileFromSparseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingFileFromSparseCharFeatures) StreamingFileFromSparseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingFileFromSparseByteFeatures) StreamingFileFromSparseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingFileFromSparseShortFeatures) StreamingFileFromSparseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingFileFromSparseWordFeatures) StreamingFileFromSparseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingFileFromSparseIntFeatures) StreamingFileFromSparseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingFileFromSparseUIntFeatures) StreamingFileFromSparseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingFileFromSparseLongFeatures) StreamingFileFromSparseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingFileFromSparseUlongFeatures) StreamingFileFromSparseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingFileFromSparseShortRealFeatures) StreamingFileFromSparseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingFileFromSparseRealFeatures) StreamingFileFromSparseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingFileFromSparseLongRealFeatures) StreamingFileFromSparseFeatures<floatmax_t>;
#endif
}

/* Template Class StreamingFileFromDenseFeatures */
%include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>
namespace shogun
{
#ifdef USE_BOOL
    %template(StreamingFileFromBoolFeatures) StreamingFileFromDenseFeatures<bool>;
#endif
#ifdef USE_CHAR
    %template(StreamingFileFromCharFeatures) StreamingFileFromDenseFeatures<char>;
#endif
#ifdef USE_UINT8
    %template(StreamingFileFromByteFeatures) StreamingFileFromDenseFeatures<uint8_t>;
#endif
#ifdef USE_INT16
    %template(StreamingFileFromShortFeatures) StreamingFileFromDenseFeatures<int16_t>;
#endif
#ifdef USE_UINT16
    %template(StreamingFileFromWordFeatures) StreamingFileFromDenseFeatures<uint16_t>;
#endif
#ifdef USE_INT32
    %template(StreamingFileFromIntFeatures) StreamingFileFromDenseFeatures<int32_t>;
#endif
#ifdef USE_UINT32
    %template(StreamingFileFromUIntFeatures) StreamingFileFromDenseFeatures<uint32_t>;
#endif
#ifdef USE_INT64
    %template(StreamingFileFromLongFeatures) StreamingFileFromDenseFeatures<int64_t>;
#endif
#ifdef USE_UINT64
    %template(StreamingFileFromUlongFeatures) StreamingFileFromDenseFeatures<uint64_t>;
#endif
#ifdef USE_FLOAT32
    %template(StreamingFileFromShortRealFeatures) StreamingFileFromDenseFeatures<float32_t>;
#endif
#ifdef USE_FLOAT64
    %template(StreamingFileFromRealFeatures) StreamingFileFromDenseFeatures<float64_t>;
#endif
#ifdef USE_FLOATMAX
    %template(StreamingFileFromLongRealFeatures) StreamingFileFromDenseFeatures<floatmax_t>;
#endif
}

%include <shogun/io/CSVFile.h>
%include <shogun/io/LibSVMFile.h>
%include <shogun/io/streaming/StreamingAsciiFile.h>
%include <shogun/io/serialization/Serializer.h>
%include <shogun/io/serialization/Deserializer.h>
%include <shogun/io/serialization/BitserySerializer.h>
%include <shogun/io/serialization/BitseryDeserializer.h>
%include <shogun/io/serialization/JsonSerializer.h>
%include <shogun/io/serialization/JsonDeserializer.h>
%include <shogun/io/stream/ByteArrayInputStream.h>
%include <shogun/io/stream/ByteArrayOutputStream.h>

%include <shogun/io/BinaryFile.h>
%include <shogun/io/HDF5File.h>

%include <shogun/io/SimpleFile.h>
%include <shogun/io/MemoryMappedFile.h>
