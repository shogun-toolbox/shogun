/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Gil Hoben, Heiko Strathmann, Sergey Lisitsyn
 */

/* base includes required by any module */
%include "stdint.i"
%include "std_string.i"
%include "exception.i"

%feature("ref")   shogun::CSGObject "SG_REF($this);"
%feature("unref") shogun::CSGObject "SG_UNREF($this);"

#ifdef SWIGJAVA
%typemap(javainterfaces) shogun::CSGObject "java.io.Externalizable"

%typemap(javaimports) shogun::CSGObject
%{
import org.shogun.JsonSerializer;
import org.shogun.JsonDeserializer;
import org.shogun.ByteArrayOutputStream;
import org.shogun.ByteArrayInputStream;
import java.lang.StringBuffer;
import org.jblas.*;
%}
%typemap(javacode) shogun::CSGObject
%{
public void writeExternal(java.io.ObjectOutput out) throws java.io.IOException {
    ByteArrayOutputStream byteArrayOS = new ByteArrayOutputStream();
    JsonSerializer jsonSerializer = new JsonSerializer();
    jsonSerializer.attach(byteArrayOS);
    jsonSerializer.write(this);

    String obj_serialized = byteArrayOS.as_string();
    out.write(obj_serialized.getBytes());
}

public void readExternal(java.io.ObjectInput in) throws java.io.IOException, java.lang.ClassNotFoundException {
    StringBuffer sb = new StringBuffer();
    int ch;
    while ((ch=in.read()) != -1) {
        sb.append((char)ch);
    }
    ByteArrayInputStream bis = new ByteArrayInputStream(sb.toString());
    JsonDeserializer jsonDeserializer = new JsonDeserializer();
    jsonDeserializer.attach(bis);
    this.deserialize(jsonDeserializer);
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

 #include <shogun/base/init.h>
 #include <shogun/lib/common.h>
 #include <shogun/io/SGIO.h>
 #include <shogun/lib/exception/ShogunException.h>
 #include <shogun/lib/DataType.h>
 #include <shogun/base/Version.h>
 #include <shogun/base/Parallel.h>
 #include <shogun/base/SGObject.h>
 #include <shogun/lib/StoppableSGObject.h>

 extern void sg_global_print_message(FILE* target, const char* str);
 extern void sg_global_print_warning(FILE* target, const char* str);
 extern void sg_global_print_error(FILE* target, const char* str);

#ifdef SWIGR
 #include <Rdefines.h>
#endif

#ifdef SWIGPYTHON

 #include <shogun/io/serialization/BitserySerializer.h>
 #include <shogun/io/serialization/BitseryDeserializer.h>
 #include <shogun/io/serialization/JsonSerializer.h>
 #include <shogun/io/serialization/JsonDeserializer.h>
 #include <shogun/io/stream/ByteArrayInputStream.h>
 #include <shogun/io/stream/ByteArrayOutputStream.h>

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
                &sg_global_print_error);
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
            std::string s = obj->to_string();
            fprintf(f, "%s", s.c_str());
            return 0;
        }
%}

%feature("nothread") _swig_monkey_patch;
%feature("docstring", "Adds a Python object (such as a function) \n"
					  "to a class (method) or to a module. \n"
					  "If the name of the function conflicts with \n"
	   				  "another Python object in the same scope\n"
                      "raises a TypeError.") _swig_monkey_patch;

// taken from https://github.com/swig/swig/issues/723#issuecomment-230178855
%typemap(out) void _swig_monkey_patch "$result = PyErr_Occurred() ? NULL : SWIG_Py_Void();"
%inline %{
	static void _swig_monkey_patch(PyObject *type, PyObject *name, PyObject *object) {
		PyObject *dict = NULL;
		if (!PyUnicode_Check(name))
			{
				PyErr_SetString(PyExc_TypeError, "name is not a string");
				return;
			}

		if (PyType_Check(type)) {
			PyTypeObject *pytype = (PyTypeObject *)type;
			dict = pytype->tp_dict;
		}
		else if (PyModule_Check(type)) {
			dict = PyModule_GetDict(type);
		}
		else {
			PyErr_SetString(PyExc_TypeError, "type is not a Python type or module");
			return;
		}
		if (PyDict_Contains(dict, name))
		{
			PyErr_SetString(PyExc_ValueError, "function name already exists in the given scope");
			return;
		}
		PyDict_SetItem(dict, name, object);

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
    std::string s = obj->to_string();
    fprintf(f, "%s", s.c_str());
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
        SWIG_exception(SWIG_SystemError, const_cast<char*>(e.what()));
#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
        SWIG_fail;
#endif
    }
    SWIG_CATCH_STDEXCEPT
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
%include <shogun/lib/exception/ShogunException.h>

%include "swig_typemaps.i"

%include "std_vector.i"
namespace std {
    %template(IntStdVector) vector<int32_t>;
    %template(DoubleStdVector) vector<float64_t>;
    %template(StringStdVector) vector<string>;
}

#ifndef SWIGR
%include <shogun/base/init.h>
#endif
%include <shogun/base/SGObject.h>
%include <shogun/io/SGIO.h>
%include <shogun/base/Version.h>
%include <shogun/base/Parallel.h>
%include <shogun/lib/StoppableSGObject.h>

namespace shogun
{
    %extend CSGObject
    {
        std::vector<std::string> parameter_names() const {
            std::vector<std::string> result;
            for (auto const& each: $self->get_params()) {
                result.push_back(each.first);
            }
            return result;
        }

        std::string parameter_type(const std::string& name) const {
            auto params = $self->get_params();
            if (params.find(name) != params.end()) {
                return params[name].get()->get_value().type();
            }
            else {
                SG_SERROR("There is no parameter called '%s' in %s", name.c_str(), $self->get_name());
            }
        }

        bool parameter_is_sg_base(const std::string& name) const {
            auto params = $self->get_params();
            if (params.find(name) != params.end()) {
                if ($self->get(name, std::nothrow) != nullptr)
                    return true;
                else
                    return false;
            }
            else
            {
                SG_SERROR("There is no parameter called '%s' in %s", name.c_str(), $self->get_name());
            }
        }

        std::vector<std::string> param_options(const std::string& name) const {
            std::vector<std::string> result;

            auto param_to_enum_map = $self->get_string_to_enum_map();

            if (param_to_enum_map.find(name) == param_to_enum_map.end())
            {
                SG_SERROR("There are no options for parameter %s::%s", $self->get_name(), name.c_str());
            }

            auto string_to_enum_map = param_to_enum_map[name];

            for (auto const& each: string_to_enum_map)
                result.push_back(each.first);

            return result;
        }

#ifdef SWIGPYTHON
        std::string __str__() const
        {
            return $self->to_string();
        }

        std::string __repr__() const
        {
            return $self->to_string();
        }

        PyObject* __reduce_ex__(int proto)
        {
            pickle_ascii = (proto==0) ? 1 : 0;
            Py_RETURN_NONE;
        }

        PyObject* __getstate__()
        {
            io::CSerializer* serializer = nullptr;
            if (pickle_ascii)
                serializer = new io::CJsonSerializer();
            else
                serializer = new io::CBitserySerializer();
            auto byte_stream = some<io::CByteArrayOutputStream>();
            serializer->attach(byte_stream);
            serializer->write(wrap($self));

            auto serialized_obj = byte_stream->content();
            SG_UNREF(serializer);
#ifdef PYTHON3
            PyObject* str=PyBytes_FromStringAndSize(serialized_obj.data(), serialized_obj.size());
#else
            PyObject* str=PyString_FromStringAndSize(serialized_obj.data(), serialized_obj.size());
#endif
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
            io::CDeserializer* deser = nullptr;
            if (pickle_ascii)
                deser = new io::CJsonDeserializer();
            else
                deser = new io::CBitseryDeserializer();

            auto byte_input_stream = some<io::CByteArrayInputStream>(str, len);
            deser->attach(byte_input_stream);
            $self->deserialize(deser);
            SG_UNREF(deser);
        }

        /*int getbuffer(PyObject *obj, Py_buffer *view, int flags) { return 0; }*/
#endif //SWIGPYTHON
    }
}

#ifdef SWIGPYTHON
%pythoncode %{
try:
    import copy_reg
except ImportError:
    import copyreg as copy_reg
def _sg_reconstructor(cls, base, state):
    try:
        if isinstance(cls, str) and cls.startswith('shogun.'):
            if base is object:
                import shogun
                return eval(cls+'()')
            else:
                base.__new__(cls, state)
                if base.__init__ != object.__init__:
                    base.__init__(obj, state)
            return obj
        if isinstance(cls(), SGObject):
            if base is object:
                 obj = cls()
            else:
                obj = base.__new__(cls, state)
                if base.__init__ != object.__init__:
                    base.__init__(obj, state)
            return obj

        return _py_orig_reconstructor(cls, base, state)
    except:
        return _py_orig_reconstructor(cls, base, state)

def _sg_reduce_ex(self, proto):
    try:
        if not isinstance(self, SGObject):
            return _py_orig_reduce_ex(self, proto)
    except:
        return _py_orig_reduce_ex(self, proto)

    base = object
    state = None
    args = ('shogun.' + self.get_name(), base, state)


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
