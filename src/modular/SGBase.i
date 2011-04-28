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




%include stl.i
/* instantiate the required template specializations */
namespace std {
  %template(IntVector)    vector<int32_t>;
  %template(DoubleVector) vector<float64_t>;
  %template(StringVector) vector<string>;
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
