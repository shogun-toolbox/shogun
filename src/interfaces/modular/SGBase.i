/* base includes required by any module */
%include "stdint.i"
%include "exception.i"

#ifdef SWIGJAVA
%typemap(javainterfaces) shogun::CSGObject "java.io.Externalizable"

%typemap(javaimports) shogun::CSGObject
%{
import org.shogun.SerializableFile;
import org.shogun.SerializableAsciiFile;
%}
%typemap(javacode) shogun::CSGObject
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
#if defined(SWIGPERL) && defined(HAVE_PDL)
#ifdef __cplusplus
  extern "C" {
#endif
#include <pdlcore.h>

#include <ppport.h>

#ifdef __cplusplus
  }
#endif
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
 #include <shogun/base/SGRefObject.h>
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

#ifdef SWIGPYTHON

 #include <shogun/io/SerializableFile.h>
 #include <shogun/io/SerializableAsciiFile.h>
 #include <shogun/io/SerializableHdf5File.h>

 static int pickle_ascii;
#endif

 using namespace shogun;
%}

#if  defined (SWIGPERL) && defined(HAVE_PDL)
%header %{
  SV* CoreSV;
  Core* PDL;
%}
#endif

%init %{

#if  defined (SWIGPERL) && defined(HAVE_PDL)
  //check Core.xs //Core* PDL_p = pdl__Core_get_Core();
  //PDL_COMMENT("Get pointer to structure of core shared C routines")
  //PDL_COMMENT("make sure PDL::Core is loaded")

  perl_require_pv("PDL::Core");
  CoreSV = perl_get_sv("PDL::SHARE",FALSE);
  //  PDL_COMMENT("SV* value")
  if (CoreSV == NULL)
    Perl_croak(aTHX_ "Can't load PDL::Core module");
  PDL = INT2PTR(Core*, SvIV( CoreSV ));
  //  PDL_COMMENT("Core* value")
#endif


#ifdef SWIGPYTHON
        import_array();
#endif

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
#ifndef DISABLE_CANCEL_CALLBACK
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error, &sg_global_cancel_computations);
#else
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error);
#endif
#endif


#ifdef SWIGRUBY
        rb_require("narray");
        cNArray = rb_const_get(rb_cObject, rb_intern("NArray"));

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

#ifdef SWIGPYTHON
%{
        static int print_sgobject(PyObject *pyobj, FILE *f, int flags) {
            void *argp;
            int res;
            res = SWIG_ConvertPtr(pyobj, &argp, SWIGTYPE_p_shogun__CSGObject, 0);
            if (!SWIG_IsOK(res)) {
                SWIG_Error(SWIG_ArgError(res), "in method 'CSGObject::tp_print', argument 1 of type 'CSGObject *'");
                return SWIG_ERROR;
            }

            CSGObject *obj = reinterpret_cast<CSGObject*>(argp);
            fprintf(f, "%s", obj->get_name());
            return 0;
        }
%}

%typemap(out) PyObject* __reduce_ex__(int proto)
{
    return PyObject_CallMethod(self, (char*) "__reduce__", (char*) "");
}

%typemap(in) __setstate__(PyObject* state) {
    $1 = $input;
}

%typemap(out) PyObject* __getstate__()
{
    $result=$1;
}

#elif defined(SWIGPERL)
%{
  static int print_sgobject(SV* pobj, FILE *f, int flags) {
    void *argp;
    int res;
    res = SWIG_ConvertPtr(pobj, &argp, SWIGTYPE_p_shogun__CSGObject, 0);
    if (!SWIG_IsOK(res)) {
      SWIG_Error(SWIG_ArgError(res), "in method 'CSGObject::tp_print', argument 1 of type 'CSGObject *'");
      return SWIG_ERROR;
    }
    CSGObject *obj = reinterpret_cast<CSGObject*>(argp);
    fprintf(f, "%s", obj->get_name());
    return 0;
  }
%}
#endif

%exception
{
    try
    {
        $action
    }
#if defined(SWIGPYTHON) && defined(USE_SWIG_DIRECTORS)
    catch (Swig::DirectorException &e)
    {
        SWIG_fail;
    }
#endif
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


%rename(SGObject) CSGObject;

%include <shogun/lib/common.h>

%include "swig_typemaps.i"

#if !defined(SWIGJAVA)
%include "std_vector.i"
namespace std {
    %template(IntStdVector) vector<int32_t>;
    %template(DoubleStdVector) vector<float64_t>;
}
#endif

#ifndef SWIGR
%include <shogun/base/init.h>
#endif
%include <shogun/base/SGRefObject.h>
%include <shogun/base/SGObject.h>
%include <shogun/io/SGIO.h>
%include <shogun/base/Version.h>
%include <shogun/base/Parallel.h>

%feature("ref")   SGRefObject "SG_REF($this);"
%feature("unref") SGRefObject "SG_UNREF($this);"

#ifdef SWIGPYTHON
namespace shogun
{

    %extend CSGObject
    {
        const char* __str__() const
        {
            return $self->get_name();
        }

        const char* __repr__() const
        {
            return $self->get_name();
        }

        PyObject* __reduce_ex__(int proto)
        {
            pickle_ascii= (proto==0) ? 1 : 0;
            return NULL;
        }

        PyObject* __getstate__()
        {
            char* fname=tmpnam(NULL);
            FILE* tmpf=fopen(fname, "w");
            CSerializableFile* fstream=NULL;
#ifdef HAVE_HDF5
            if (pickle_ascii)
                fstream = new CSerializableAsciiFile(fname, 'w');
            else
                fstream = new CSerializableHdf5File(fname, 'w');
#else
            fstream = new CSerializableAsciiFile(fname, 'w');
#endif
            $self->save_serializable(fstream);
            fstream->close();
            delete fstream;

            size_t len=0;
            char* result=CFile::read_whole_file(fname, len);
            unlink(fname);

#ifdef PYTHON3
            PyObject* str=PyBytes_FromStringAndSize(result, len);
#else
            PyObject* str=PyString_FromStringAndSize(result, len);
#endif
            SG_FREE(result);

            PyObject* tuple=PyTuple_New(2);
            PyTuple_SetItem(tuple, 0, PyBool_FromLong(pickle_ascii));
            PyTuple_SetItem(tuple, 1, str);
            return tuple;
        }


        void __setstate__(PyObject* state)
        {
            PyObject* py_ascii = PyTuple_GetItem(state,0);
            pickle_ascii= (py_ascii == Py_True) ? 1 : 0;
            PyObject* py_str = PyTuple_GetItem(state,1);
            char* str=NULL;
            Py_ssize_t len=0;

#ifdef PYTHON3
            PyBytes_AsStringAndSize(py_str, &str, &len);
#else
            PyString_AsStringAndSize(py_str, &str, &len);
#endif

            char* fname=tmpnam(NULL);
            FILE* tmpf=fopen(fname, "w");;
            size_t total = fwrite(str, (size_t) 1, (size_t) len, tmpf);
            fclose(tmpf);
            ASSERT(total==len);

            CSerializableFile* fstream=NULL;
#ifdef HAVE_HDF5
            if (pickle_ascii)
                fstream = new CSerializableAsciiFile(fname, 'r');
            else
            {
                try
                {
                    fstream = new CSerializableHdf5File(fname, 'r');
                }
                catch (ShogunException& e)
                {
                    fstream = new CSerializableAsciiFile(fname, 'r');
                }
            }
#else
            try
            {
                fstream = new CSerializableAsciiFile(fname, 'r');
            }
            catch (ShogunException& e)
            {
                    SG_SERROR("File contains an HDF5 stream but " \
                            "Shogun was not compiled with HDF5" \
                            " support! -  cannot load file %s." \
                            " (exception was %s)", e.get_exception_string());
            }
#endif
            $self->load_serializable(fstream);
            fstream->close();
            delete fstream;
            unlink(fname);
        }

        /*int getbuffer(PyObject *obj, Py_buffer *view, int flags) { return 0; }*/
    }
}

%pythoncode %{
try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg
def _sg_reconstructor(cls, base, state):
    try:
        if not isinstance(cls(), SGObject):
            return _py_orig_reconstructor(cls, base, state)
    except:
        return _py_orig_reconstructor(cls, base, state)

    if base is object:
        obj = cls() #object.__new__(cls)
    else:
        obj = base.__new__(cls, state)
        if base.__init__ != object.__init__:
            base.__init__(obj, state)
    return obj


def _sg_reduce_ex(self, proto):
    try:
        if not isinstance(self, SGObject):
            return _py_orig_reduce_ex(self, proto)
    except:
        return _py_orig_reduce_ex(self, proto)

    base = object # not really reachable
    if base is object:
        state = None
    else:
        if base is self.__class__:
            raise TypeError("can't pickle %s objects" % base.__name__)
        state = base(self)
    args = (self.__class__, base, state)
    try:
        getstate = self.__getstate__
    except AttributeError:
        if getattr(self, "__slots__", None):
            raise TypeError("a class that defines __slots__ without "
                            "defining __getstate__ cannot be pickled")
        try:
            dict = self.__dict__
        except AttributeError:
            dict = None
    else:
        dict = getstate()
    if dict:
        return _sg_reconstructor, args, dict
    else:
        return _sg_reconstructor, args

_py_orig_reduce_ex=copy_reg._reduce_ex
_py_orig_reconstructor=copy_reg._reconstructor

copy_reg._reduce_ex=_sg_reduce_ex
copy_reg._reconstructor=_sg_reconstructor
%}

#endif /* SWIGPYTHON  */
