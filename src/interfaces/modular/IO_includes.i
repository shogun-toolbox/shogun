%{
#include <shogun/io/IOBuffer.h>
#include <shogun/io/ParseBuffer.h>
#include <shogun/io/InputParser.h>
#include <shogun/io/File.h>
#include <shogun/io/StreamingFile.h>
#include <shogun/io/StreamingFileFromFeatures.h>
#include <shogun/io/StreamingFileFromSparseFeatures.h>
#include <shogun/io/StreamingFileFromSimpleFeatures.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/classifier/vw/VwParser.h>
#include <shogun/io/StreamingVwFile.h>
#include <shogun/io/StreamingVwCacheFile.h>
#include <shogun/io/BinaryFile.h>
#include <shogun/io/HDF5File.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableHdf5File.h>
#include <shogun/io/SerializableJsonFile.h>
#include <shogun/io/SerializableXmlFile.h>
#include <shogun/io/SimpleFile.h>
#include <shogun/io/MemoryMappedFile.h>
%}

