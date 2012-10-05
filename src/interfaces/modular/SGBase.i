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

#ifdef SWIGRUBY
%{
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
%}
#endif
#ifdef SWIGPERL
#ifdef HAVE_PDL
%{
#ifdef __cplusplus
  extern "C" {
#endif
#include <pdlcore.h>

#include <ppport.h>

#ifdef __cplusplus
  }
#endif
%}
#endif
#endif

%{
 /* required for python */
 #define SWIG_FILE_WITH_INIT
%}
#if defined(SWIGJAVA) || defined(SWIGCSHARP)
%{
 #include <shogun/base/init.h>
%}
#endif
%{
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
%}

#ifdef SWIGR
%{
 #include <Rdefines.h>
%}
#endif

#if defined(SWIGPYTHON) || defined (SWIGPERL)
%{
 #include <shogun/io/SerializableFile.h>
 #include <shogun/io/SerializableAsciiFile.h>
 #include <shogun/io/SerializableHdf5File.h>

 static int pickle_ascii;
%}
#endif

%{
 using namespace shogun;
%}


#if defined(SWIGPYTHON) || defined (SWIGPERL)
%init %{
  import_array();
%}
#endif

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
%init %{
#ifndef DISABLE_CANCEL_CALLBACK
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error, &sg_global_cancel_computations);
#else
        shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
                &sg_global_print_error);
#endif
%}
#endif
#if  defined (SWIGPERL) //&& HAVE_PDL

%header %{
  SV* CoreSV;
  Core* PDL;
%}

%init %{
  //PTZ120930 boot PDL, load PDL stuff??? and define a PDL
  //pl_require('PDL');
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

#if !defined(XS_VERSION)
#define XS_VERSION "NAN"
#endif

  if (PDL->Version != PDL_CORE_VERSION)
    Perl_croak(aTHX_ "[PDL->Version: %d PDL_CORE_VERSION: %d XS_VERSION: %s] PDL::Bad needs to be recompiled against the newly installed PDL", PDL->Version, PDL_CORE_VERSION, XS_VERSION);


#if 0
BOOT:

   PDL_COMMENT("Get pointer to structure of core shared C routines")
   PDL_COMMENT("make sure PDL::Core is loaded")
   perl_require_pv("PDL::Core");
   CoreSV = perl_get_sv("PDL::SHARE",FALSE);  PDL_COMMENT("SV* value")
   if (CoreSV==NULL)
     Perl_croak(aTHX_ "Can't load PDL::Core module");
   PDL = INT2PTR(Core*, SvIV( CoreSV ));  PDL_COMMENT("Core* value")
   if (PDL->Version != PDL_CORE_VERSION)
     Perl_croak(aTHX_ "[PDL->Version: %d PDL_CORE_VERSION: %d XS_VERSION: %s] PDL::Bad needs to be recompiled against the newly installed PDL", PDL->Version, PDL_CORE_VERSION, XS_VERSION);
   //_nan_float = __pdl_nan.__d;
   //_nan_double = (double) __pdl_nan.__d;
           
#endif
%}
#endif


#ifdef SWIGRUBY
%init %{
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
%}
#endif



#if defined(SWIGPYTHON)

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

#elseif defined (SWIGPERL)

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
//PTZ120924 whao...need to try harder...?

%typemap(out) SV* __reduce_ex__(int proto)
{
  //TPZ120926 not good!!use stack spare..

  return SWIG_CALLXS("__reduce__");

    //return PyObject_CallMethod(self, (char*) "__reduce__", (char*) "");
}

%typemap(in) __setstate__(SV* state) {
    $1 = $input;
}       

%typemap(out) SV* __getstate__()
{
    $result = $1;
}
#endif




%exception
{
    try
    {
        $action
    }
#if (defined(SWIGPYTHON) || defined(SWIGPERL)) && defined(USE_SWIG_DIRECTORS)
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

%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"

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
%include <shogun/io/SGIO.h>
SERIALIZABLE_DUMMY(shogun::SGIO);

%include <shogun/base/SGObject.h>
%include <shogun/base/Version.h>
SERIALIZABLE_DUMMY(shogun::Version);
%include <shogun/base/Parallel.h>
SERIALIZABLE_DUMMY(shogun::Parallel);

#if defined(SWIGPYTHON)

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
                fstream = new CSerializableHdf5File(fname, 'r');
#else
            if (!pickle_ascii)
                SG_SERROR("File contains an HDF5 stream but " \
                        "Shogun was not compiled with HDF5" \
                        " support! -  cannot load.");
            fstream = new CSerializableAsciiFile(fname, 'r');
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

/* SWIGPYTHON  */
#elseif defined (SWIGPERL)

/*%todo
 */
//typedef pdl pdl;

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

        SV* __reduce_ex__(int proto)
        {
            pickle_ascii = (proto == 0) ? 1 : 0;
            return NULL;
        }

	//XS(XS_shogun___getstate__);
	//SV** __getstate__()
	XS(XS_Shogun_CSGObject___getstate__);
	SV**  __getstate__()
        {
#ifdef dVAR
	  dVAR; dXSARGS;
#else
	  dXSARGS;
#endif
	  dSP;
	  if (items != 1)
	    croak_xs_usage(cv,  "CSGObject, ...");
	  {
            char* fname = tmpnam(NULL);
            FILE* tmpf = fopen(fname, "w");
            CSerializableFile* fstream = NULL;
            size_t len = 0;

#if 0
	    STMT_START {
	      SV* const xsub_tmp_sv = ST(0);
	      SvGETMAGIC(xsub_tmp_sv);
	      if (SvROK(xsub_tmp_sv) && SvTYPE(SvRV(xsub_tmp_sv)) == SVt_PVHV){
		s = (HV*)SvRV(xsub_tmp_sv);
	      }else{
		Perl_croak(aTHX_ "%s: %s is not a HASH reference",
			   "Shogun::CSGObject::__getstate__",
			   "s");
	      }
	    } STMT_END;
#endif

#ifdef HAVE_HDF5
            if (pickle_ascii)
                fstream = new CSerializableAsciiFile(fname, 'w');
            else
                fstream = new CSerializableHdf5File(fname, 'w');
#else
            fstream = new CSerializableAsciiFile(fname, 'w');
#endif
	    RETVAL = pickle_ascii;
	    XPUSHu((UV) RETVAL);

            $self->save_serializable(fstream);
            fstream->close();
            delete fstream;
            char* result = CFile::read_whole_file(fname, len);
            unlink(fname);

	    SV* pl_str= sv_2mortal(newSVpv(result,(STRLEN) len));
            SG_FREE(result);
	    XPUSHs(pl_str);
	  }
	  XSRETURN(2);
        }
	
	//XS(XS_Shogun_CSGObject___setstate__)
        void __setstate__(U32  pl_ascii, char *pl_str, STRLEN pl_len)
        {
#ifdef dVAR
	  dVAR;
#endif
	  dXSARGS;
	  if (items != 4)
	    croak_xs_usage(cv, "CSGObject,  ascii_flag, string");
	  if (items == 4)
	    {
#if 0	      
	      CSGObject* pn;
	      if (SvROK(ST(0)) && sv_derived_from(ST(0), "CSGObjectPtr")) {
		IV tmp = SvIV((SV*)SvRV(ST(0)));
		pn = INT2PTR(CSGObject *,tmp);
	      }
	      else
		Perl_croak(aTHX_ "%s: %s is not of type %s",
			   "Shogun::CSGObjectPtr::__setstate__",
			   "pn", "CSGObjectPtr");



	      SV* pl_ascii = SvREFCNT_inc(ST(1));
#endif
	      pickle_ascii = (SvTRUE(pl_ascii) ? 1 : 0); //const U32 flags SVf_UTF8
	      //could also use  bool    is_ascii_string(const U8 *s, STRLEN len)

	      //SV* pl_str = SvPV_nolen(ST(2));
	      //char* str = NULL;
	      //newSVpvn_flags((s), (len), (u) ? 0 : SVf_UTF8)
	      //SV*     newSVpvn_flags(const char *const s, const STRLEN len, const U32 flags);

	      //PTZ120923dump my string into a temp file and then stream it whao....
	      char* fname = tmpnam(NULL);
	      FILE* tmpf = fopen(fname, "w");
	      size_t total = fwrite(pl_str, (size_t) 1, (size_t) pl_len, tmpf);
	      fclose(tmpf);
	      ASSERT(total == len);

	      CSerializableFile* fstream=NULL;

	      //PTZ120924 obviously here "pickle_ascii" is used to check with data format!!! how wondefull.
#ifdef HAVE_HDF5
	      if (pickle_ascii)
                fstream = new CSerializableAsciiFile(fname, 'r');
	      else
                fstream = new CSerializableHdf5File(fname, 'r');
#else
	      if (!pickle_ascii)
                SG_SERROR("File contains an HDF5 stream but "	\
			  "Shogun was not compiled with HDF5"	\
			  " support! -  cannot load.");
	      fstream = new CSerializableAsciiFile(fname, 'r');
#endif
	      $self->load_serializable(fstream);
	      fstream->close();
	      delete fstream;
	      unlink(fname);
	      //PTZ120924 much ado about nothing, might be better of with an IPC around
	    }
	}
        /*int getbuffer(PyObject *obj, Py_buffer *view, int flags) { return 0; }*/
    }
}

%}

%perlcode %{

  //PTZ120930 boot pdl ?


  eval {use Copy_Reg;};
  if($@) {
    package Copy_Reg {
      use base(qw(CopyReg));
    };
  }

  sub _sg_reconstructor
  {
    my ($self, $cls, $base, $state) = @_;
  try:
    eval {
      if(not isinstance($cls->(), SGObject)) {
	return _py_orig_reconstructor($cls, $base, $state);
      }
    };
  except:
    if($@) {
      return $self->_pl_orig_reconstructor($cls, $base, $state);
    }
    my $obj;
    if(ref($base)) {
      $obj = $cls->();
    } else {
      $obj = $base->new($cls, $state);
      if($base->__init__() != $object->__init__()) {
	$base->__init__($obj, $state);
      }
    }
    return $obj;
  }

  sub _sg_reduce_ex
  {
    my ($self, $proto) = @_;
    eval {
      if(not isinstance($self, $SGObject)) {
	return &_pl_orig_reduce_ex($self, $proto);
      }
    };
    if($@) {
      return &_pl_orig_reduce_ex($self, $proto);
    }
    my $base = $object;
    if($base is $object) {
      $state = 'None';
    }
    else {
      if( $base is $self->__class__) {
	croack("can't pickle %s objects", $base->__name__);
      }
      $state = base($self);
    }
    my @args = ($self->__class__, base, state);
    eval {
      getstate = $self->__getstate__;
    }
    except AttributeError:
        if getattr(self, "__slots__", None):
            raise TypeError("a class that defines __slots__ without "
                            "defining __getstate__ cannot be pickled")

	    eval {
	      $dict = $self->__dict__;
	      }
        except AttributeError:
	if($@) {
	my $dict;
	} else {
	      my $dict = $self->getstate();
	      if($dict){
		return(\&_sg_reconstructor, $args, $dict);
	      } else {
		return(\&_sg_reconstructor, $args};
	      }
_py_orig_reduce_ex=copy_reg._reduce_ex
_py_orig_reconstructor=copy_reg._reconstructor
      
copy_reg._reduce_ex=_sg_reduce_ex
copy_reg._reconstructor=_sg_reconstructor


%}


#endif /* SWIGPERL  */
