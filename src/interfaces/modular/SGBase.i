/* base includes required by any module */
%include "stdint.i"
%include "exception.i"
%include "std_string.i"

%{
 /* required for python */
 #define SWIG_FILE_WITH_INIT

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
 #include <shogun/base/init.h>
#endif
 #include <shogun/lib/common.h>
 #include <shogun/lib/io.h>
 #include <shogun/lib/ShogunException.h>
 #include <shogun/lib/DataType.h>
 #include <shogun/base/Version.h>
 #include <shogun/base/Parallel.h>
 #include <shogun/base/SGObject.h>
 #include <shogun/base/DynArray.h>

 extern void sg_global_print_message(FILE* target, const char* str);
 extern void sg_global_print_warning(FILE* target, const char* str);
 extern void sg_global_print_error(FILE* target, const char* str);
#ifndef DISABLE_CANCEL_CALLBACK
 extern void sg_global_cancel_computations(bool &delayed, bool &immediately);
#endif

#ifdef SWIGR
 #include <Rdefines.h>
#endif

 using namespace shogun;
%}

/*#ifdef SWIGJAVA
%pragma(java) moduleimports=%{
    import java.io.*; // For Serializable
%}
%pragma(java) jniclassinterfaces="Serializable"
%pragma(java) moduleinterfaces="Serializable"
#endif
*/

%init %{
#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
#ifndef DISABLE_CANCEL_CALLBACK
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error, &sg_global_cancel_computations);
#else
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error);
#endif
#endif

#ifdef SWIGPYTHON
        import_array();
#endif
%}

%exception
{
    try
    {
        $action
    }
    catch (std::bad_alloc)
    {
        SWIG_exception(SWIG_MemoryError, const_cast<char*>("Out of memory error.\n"));
#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
        SWIG_fail;
#endif
    }
    catch (shogun::ShogunException e)
    {
        SWIG_exception(SWIG_SystemError, const_cast<char*>(e.get_exception_string()));
#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
        SWIG_fail;
#endif
    }
}

%ignore NUM_LOG_LEVELS;
%ignore FBUFSIZE;
/* %ignore init_shogun;
%ignore exit_shogun; */
%ignore sg_print_message;
%ignore sg_print_warning;
%ignore sg_print_error;
%ignore sg_cancel_computations;

%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"

%rename(SGObject) CSGObject;

%include <shogun/lib/common.h>

%include "swig_typemaps.i"

#ifndef SWIGR
%include <shogun/base/init.h>
#endif
%include <shogun/lib/ShogunException.h>
%include <shogun/lib/io.h>
%include <shogun/base/SGObject.h>
%include <shogun/base/Version.h>
%include <shogun/base/Parallel.h>
%include <shogun/lib/DataType.h>

namespace shogun
{
    %template(BoolSparseMatrix) SGSparseMatrix<bool>;
    %template(CharSparseMatrix) SGSparseMatrix<char>;
    %template(ByteSparseMatrix) SGSparseMatrix<uint8_t>;
    %template(WordSparseMatrix) SGSparseMatrix<uint16_t>;
    %template(ShortSparseMatrix) SGSparseMatrix<int16_t>;
    %template(IntSparseMatrix)  SGSparseMatrix<int32_t>;
    %template(UIntSparseMatrix)  SGSparseMatrix<uint32_t>;
    %template(LongIntSparseMatrix)  SGSparseMatrix<int64_t>;
    %template(ULongIntSparseMatrix)  SGSparseMatrix<uint64_t>;
    %template(ShortRealSparseMatrix) SGSparseMatrix<float32_t>;
    %template(RealSparseMatrix) SGSparseMatrix<float64_t>;
    %template(LongRealSparseMatrix) SGSparseMatrix<floatmax_t>;

    %template(BoolStringList) SGStringList<bool>;
    %template(CharStringList) SGStringList<char>;
    %template(ByteStringList) SGStringList<uint8_t>;
    %template(WordStringList) SGStringList<uint16_t>;
    %template(ShortStringList) SGStringList<int16_t>;
    %template(IntStringList)  SGStringList<int32_t>;
    %template(UIntStringList)  SGStringList<uint32_t>;
    %template(LongIntStringList)  SGStringList<int64_t>;
    %template(ULongIntStringList)  SGStringList<uint64_t>;
    %template(ShortRealStringList) SGStringList<float32_t>;
    %template(RealStringList) SGStringList<float64_t>;
    %template(LongRealStringList) SGStringList<floatmax_t>;


    %template(BoolString) SGString<bool>;
    %template(CharString) SGString<char>;
    %template(ByteString) SGString<uint8_t>;
    %template(WordString) SGString<uint16_t>;
    %template(ShortString) SGString<int16_t>;
    %template(IntString)  SGString<int32_t>;
    %template(UIntString)  SGString<uint32_t>;
    %template(LongIntString)  SGString<int64_t>;
    %template(ULongIntString)  SGString<uint64_t>;
    %template(ShortRealString) SGString<float32_t>;
    %template(RealString) SGString<float64_t>;
    %template(LongRealString) SGString<floatmax_t>;

    %template(BoolVector) SGVector<bool>;
    %template(CharVector) SGVector<char>;
    %template(ByteVector) SGVector<uint8_t>;
    %template(WordVector) SGVector<uint16_t>;
    %template(ShortVector) SGVector<int16_t>;
    %template(IntVector)  SGVector<int32_t>;
    %template(UIntVector)  SGVector<uint32_t>;
    %template(LongIntVector)  SGVector<int64_t>;
    %template(ULongIntVector)  SGVector<uint64_t>;
    %template(ShortRealVector) SGVector<float32_t>;
    %template(RealVector) SGVector<float64_t>;
    %template(LongRealVector) SGVector<floatmax_t>;

    %template(BoolMatrix) SGMatrix<bool>;
    %template(CharMatrix) SGMatrix<char>;
    %template(ByteMatrix) SGMatrix<uint8_t>;
    %template(WordMatrix) SGMatrix<uint16_t>;
    %template(ShortMatrix) SGMatrix<int16_t>;
    %template(IntMatrix)  SGMatrix<int32_t>;
    %template(UIntMatrix)  SGMatrix<uint32_t>;
    %template(LongIntMatrix)  SGMatrix<int64_t>;
    %template(ULongIntMatrix)  SGMatrix<uint64_t>;
    %template(ShortRealMatrix) SGMatrix<float32_t>;
    %template(RealMatrix) SGMatrix<float64_t>;
    %template(LongRealMatrix) SGMatrix<floatmax_t>;

    %template(BoolNDArray) SGNDArray<bool>;
    %template(CharNDArray) SGNDArray<char>;
    %template(ByteNDArray) SGNDArray<uint8_t>;
    %template(WordNDArray) SGNDArray<uint16_t>;
    %template(ShortNDArray) SGNDArray<int16_t>;
    %template(IntNDArray)  SGNDArray<int32_t>;
    %template(UIntNDArray)  SGNDArray<uint32_t>;
    %template(LongIntNDArray)  SGNDArray<int64_t>;
    %template(ULongIntNDArray)  SGNDArray<uint64_t>;
    %template(ShortRealNDArray) SGNDArray<float32_t>;
    %template(RealNDArray) SGNDArray<float64_t>;
    %template(LongRealNDArray) SGNDArray<floatmax_t>;
}



%include stl.i
/* instantiate the required template specializations */
namespace std {
  %template(IntStdVector)    vector<int32_t>;
  %template(DoubleStdVector) vector<float64_t>;
  %template(StringStdVector) vector<string>;
}

#ifdef SWIGPYTHON

%pythoncode %{
import tempfile, random, os, exceptions

try: import Library as shogunLibrary
except exceptions.ImportError: import shogun.Library as shogunLibrary

def __SGgetstate__(self):
    fname = tempfile.gettempdir() + "/" + tempfile.gettempprefix() \
        + str(random.randint(0, 1e15))

    try:
        fstream = shogunLibrary.SerializableAsciiFile(fname, "w") \
            if self.__pickle_ascii__ \
            else shogunLibrary.SerializableHDF5File(fname, "w")
    except exceptions.AttributeError:
        fstream = shogunLibrary.SerializableAsciiFile(fname, "w")
        self.__pickle_ascii__ = True

    if not self.save_serializable(fstream):
        fstream.close(); os.remove(fname)
        raise exceptions.IOError("Could not dump Shogun object!")
    fstream.close()

    fstream = open(fname, "r"); result = fstream.read();
    fstream.close()

    os.remove(fname)
    return (self.__pickle_ascii__, result)

def __SGsetstate__(self, state_tuple):
    self.__init__()

    fname = tempfile.gettempdir() + "/" + tempfile.gettempprefix() \
        + str(random.randint(0, 1e15))

    fstream = open(fname, "w"); fstream.write(state_tuple[1]);
    fstream.close()

    try:
        fstream = shogunLibrary.SerializableAsciiFile(fname, "r") \
            if state_tuple[0] \
            else shogunLibrary.SerializableHDF5File(fname, "r")
    except exceptions.AttributeError:
        raise exceptions.IOError("File contains an HDF5 stream but " \
                                 "Shogun was not compiled with HDF5" \
                                 " support!")

    if not self.load_serializable(fstream):
        fstream.close(); os.remove(fname)
        raise exceptions.IOError("Could not load Shogun object!")
    fstream.close()

    os.remove(fname)

def __SGreduce_ex__(self, protocol):
    self.__pickle_ascii__ = True if protocol == 0 else False
    return super(self.__class__, self).__reduce__()

def __SGstr__(self):
    fname = tempfile.gettempdir() + "/" + tempfile.gettempprefix() \
        + str(random.randint(0, 1e15))

    fstream = shogunLibrary.SerializableAsciiFile(fname, "w")
    if not self.save_serializable(fstream):
        fstream.close(); os.remove(fname)
        raise exceptions.IOError("Could not dump Shogun object!")
    fstream.close()

    fstream = open(fname, "r"); result = fstream.read();
    fstream.close()

    os.remove(fname)
    return result

def __SGeq__(self, other):
    return self.__str__() == str(other)

def __SGneq__(self, other):
    return self.__str__() != str(other)

SGObject.__setstate__ = __SGsetstate__
SGObject.__getstate__ = __SGgetstate__
SGObject.__reduce_ex__ = __SGreduce_ex__
SGObject.__str__ = __SGstr__
SGObject.__eq__ = __SGeq__
SGObject.__neq__ = __SGneq__
%}

#endif /* SWIGPYTHON  */
