%{
#include <shogun/io/IOBuffer.h>
#include <shogun/io/ParseBuffer.h>
#include <shogun/io/InputParser.h>
#include <shogun/lib/Cache.h>
#include <shogun/io/File.h>
#include <shogun/io/StreamingFile.h>
#include <shogun/io/StreamingFileFromFeatures.h>
#include <shogun/io/StreamingFileFromSparseFeatures.h>
#include <shogun/io/StreamingFileFromSimpleFeatures.h>
#include <shogun/io/AsciiFile.h>
#include <shogun/io/StreamingAsciiFile.h>
#include <shogun/io/BinaryFile.h>
#include <shogun/io/HDF5File.h>
#include <shogun/io/SerializableFile.h>
#include <shogun/io/SerializableAsciiFile.h>
#include <shogun/io/SerializableHdf5File.h>
#include <shogun/io/SerializableJsonFile.h>
#include <shogun/io/SerializableXmlFile.h>
#include <shogun/lib/List.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/Signal.h>
#include <shogun/io/SimpleFile.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Trie.h>
#include <shogun/io/MemoryMappedFile.h>
#include <shogun/lib/DynamicArray.h>
#include <shogun/structure/PlifBase.h>
#include <shogun/lib/Hash.h>
#include <shogun/lib/Array.h>
#include <shogun/lib/Array2.h>
#include <shogun/lib/Array3.h>
#include <shogun/lib/GCArray.h>
#include <shogun/lib/Compressor.h>
%}

