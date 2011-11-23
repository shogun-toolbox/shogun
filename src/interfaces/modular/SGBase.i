/* base includes required by any module */
%include "stdint.i"
%include "exception.i"

%define SERIALIZABLE_DUMMY(SWIGCLASS)
%extend SWIGCLASS {
bool save_serializable(CSerializableFile* file, const char* prefix="") { return false; };
bool load_serializable(CSerializableFile* file, const char* prefix="") { return false; };
}
%enddef

#ifdef SWIGJAVA
%typemap(javainterfaces) SWIGTYPE "java.io.Externalizable"

%typemap(javaincludes) SWIGTYPE
%{
import org.shogun.SerializableFile;
import org.shogun.SerializableAsciiFile;
%}
%typemap(javacode) SWIGTYPE
%{
public void writeExternal(java.io.ObjectOutput out) throws java.io.IOException {
        java.util.Random randomGenerator = new java.util.Random();
        String tmpFileName = System.getProperty("java.io.tmpdir") + "/" + randomGenerator.nextInt() + "shogun.tmp";
        java.io.File file = null; 
        java.io.FileInputStream in = null;
        int ch;
        try {
                file = new java.io.File(tmpFileName);
                file.createNewFile();
                SerializableAsciiFile tmpFile = new SerializableAsciiFile(tmpFileName, 'w');
                this.save_serializable(tmpFile);
                tmpFile.close();
                in = new java.io.FileInputStream(file);
                // TODO bufferize
                while((ch=in.read()) != -1) {
                        out.write(ch);
                }
                file.delete();
        } catch (java.io.IOException ex) {
        } finally {
                try {
                        in.close();
                } catch (java.io.IOException ex) {
                }
        }
}

public void readExternal(java.io.ObjectInput in) throws java.io.IOException, java.lang.ClassNotFoundException {
        java.util.Random randomGenerator = new java.util.Random();
        String tmpFileName = System.getProperty("java.io.tmpdir") + "/" + randomGenerator.nextInt() + "shogun.tmp";
        java.io.File file = null;
        java.io.FileOutputStream out = null;
        int ch;
        try {
                file = new java.io.File(tmpFileName);
                file.createNewFile();
                out = new java.io.FileOutputStream(file);
                while ((ch=in.read()) != -1) {
                        out.write(ch);
                }
                out.close();
                SerializableAsciiFile tmpFile = new SerializableAsciiFile(tmpFileName,'r');
                this.load_serializable(tmpFile);
                tmpFile.close();
                file.delete();
        } catch (java.io.IOException ex) {
        } finally {
                try {
                        out.close();
                } catch (java.io.IOException ex) {
                }
        }
}
    %}
#endif

%{
#ifdef SWIGRUBY
 extern "C" {
  #include <ruby.h>
  #include <narray.h>
  #include <stdlib.h>
  #include <stdio.h>
 }
 VALUE (*na_to_array_dl)(VALUE);
 VALUE (*na_to_narray_dl)(VALUE);
 VALUE cNArray;
 #include <dlfcn.h>
#endif
 /* required for python */
 #define SWIG_FILE_WITH_INIT

#if defined(SWIGJAVA) || defined(SWIGCSHARP)
 #include <shogun/base/init.h>
#endif
 #include <shogun/lib/common.h>
 #include <shogun/io/SGIO.h>
 #include <shogun/lib/ShogunException.h>
 #include <shogun/lib/DataType.h>
 #include <shogun/base/Version.h>
 #include <shogun/base/Parallel.h>
 #include <shogun/base/SGObject.h>

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

#ifdef SWIGRUBY
        extern VALUE ruby_class;
        rb_require("narray");
        cNArray = rb_const_get(ruby_class, rb_intern("NArray"));

        char* error=NULL;

        void* handle = dlopen(NARRAY_LIB, RTLD_LAZY);
        if (!handle) {
            fprintf(stderr, "%s\n", dlerror());
            exit(EXIT_FAILURE);
        }

        dlerror();    /* Clear any existing error */

        *(void **) (&na_to_array_dl) = dlsym(handle, "na_to_array");
        if ((error = dlerror()) != NULL)  {
                fprintf(stderr, "na_to_array %s\n", error);
                exit(EXIT_FAILURE);
        }

        /*if (cNArray==0)
        {
            void (*Init_narray)();
            *(void **) (&Init_narray) = dlsym(handle, "Init_narray");
            if ((error = dlerror()) != NULL)  {
                fprintf(stderr, "Init_narray %s\n", error);
                exit(EXIT_FAILURE);
            }

            fprintf(stderr, "initing narray\n");
            (*Init_narray)();
        }*/

        *(void **) (&na_to_narray_dl) = dlsym(handle, "na_to_narray");
        if ((error = dlerror()) != NULL)  {
                fprintf(stderr, "na_to_narray %s\n", error);
                exit(EXIT_FAILURE);
        }

/*        cNArray = (*(VALUE*)(dlsym(handle, "cNArray")));
        if ((error = dlerror()) != NULL)  {
                fprintf(stderr, "cNArray %s\n", error);
                exit(EXIT_FAILURE);
        }*/
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
%include <shogun/io/SGIO.h>
SERIALIZABLE_DUMMY(shogun::SGIO);
%include <shogun/base/SGObject.h>
%include <shogun/base/Version.h>
SERIALIZABLE_DUMMY(shogun::Version);
%include <shogun/base/Parallel.h>
SERIALIZABLE_DUMMY(shogun::Parallel);

#ifdef SWIGPYTHON

%pythoncode %{
import tempfile, random, os, exceptions

import modshogun

def __SGgetstate__(self):
    fname = tempfile.gettempdir() + "/" + tempfile.gettempprefix() \
        + str(random.randint(0, 1e15))

    try:
        fstream = modshogun.SerializableAsciiFile(fname, "w") \
            if self.__pickle_ascii__ \
            else modshogun.SerializableHDF5File(fname, "w")
    except exceptions.AttributeError:
        fstream = modshogun.SerializableAsciiFile(fname, "w")
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
        fstream = modshogun.SerializableAsciiFile(fname, "r") \
            if state_tuple[0] \
            else modshogun.SerializableHDF5File(fname, "r")
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

    fstream = modshogun.SerializableAsciiFile(fname, "w")
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
