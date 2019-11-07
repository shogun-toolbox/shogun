%{
#include <shogun/io/IOBuffer.h>
#include <shogun/io/streaming/ParseBuffer.h>
#include <shogun/io/streaming/InputParser.h>
#include <shogun/io/File.h>
#include <shogun/io/streaming/StreamingFile.h>
#include <shogun/io/streaming/StreamingFileFromFeatures.h>
#include <shogun/io/streaming/StreamingFileFromSparseFeatures.h>
#include <shogun/io/streaming/StreamingFileFromDenseFeatures.h>
#include <shogun/io/CSVFile.h>
#include <shogun/io/LibSVMFile.h>
#include <shogun/io/streaming/StreamingAsciiFile.h>

#include <shogun/io/BinaryFile.h>
#include <shogun/io/HDF5File.h>
#include <shogun/io/stream/InputStream.h>
#include <shogun/io/stream/OutputStream.h>
#include <shogun/io/stream/ByteArrayInputStream.h>
#include <shogun/io/stream/ByteArrayOutputStream.h>
#include <shogun/io/serialization/Serializer.h>
#include <shogun/io/serialization/Deserializer.h>
#include <shogun/io/serialization/BitserySerializer.h>
#include <shogun/io/serialization/BitseryDeserializer.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/SimpleFile.h>
#include <shogun/io/MemoryMappedFile.h>

#include <shogun/lib/Compressor.h>
%}
