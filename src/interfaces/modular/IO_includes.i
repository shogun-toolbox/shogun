%{
#include <io/IOBuffer.h>
#include <io/streaming/ParseBuffer.h>
#include <io/streaming/InputParser.h>
#include <io/File.h>
#include <io/streaming/StreamingFile.h>
#include <io/streaming/StreamingFileFromFeatures.h>
#include <io/streaming/StreamingFileFromSparseFeatures.h>
#include <io/streaming/StreamingFileFromDenseFeatures.h>
#include <io/CSVFile.h>
#include <io/LibSVMFile.h>
#include <io/streaming/StreamingAsciiFile.h>
#include <classifier/vw/VwParser.h>
#include <io/streaming/StreamingVwFile.h>
#include <io/streaming/StreamingVwCacheFile.h>
#include <io/BinaryFile.h>
#include <io/HDF5File.h>
#include <io/SerializableFile.h>
#include <io/SerializableAsciiFile.h>
#include <io/SerializableHdf5File.h>
#include <io/SerializableJsonFile.h>
#include <io/SerializableXmlFile.h>
#include <io/SimpleFile.h>
#include <io/MemoryMappedFile.h>
%}

