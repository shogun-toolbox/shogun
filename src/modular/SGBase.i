/* base includes required by any module */
%include "stdint.i"
%include "exception.i"
%include "std_string.i"

%{
 /* required for python */
 #define SWIG_FILE_WITH_INIT

 #include <shogun/base/init.h>
 #include <shogun/lib/common.h>
 #include <shogun/lib/io.h>
 #include <shogun/lib/ShogunException.h>
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

%init %{
#ifndef DISABLE_CANCEL_CALLBACK
    shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
            &sg_global_print_error, &sg_global_cancel_computations);
#else
    shogun::init_shogun(&sg_global_print_message, &sg_global_print_warning,
            &sg_global_print_error);
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
        SWIG_fail;
    }
    catch (shogun::ShogunException e)
    {
        SWIG_exception(SWIG_SystemError, const_cast<char*>(e.get_exception_string()));
        SWIG_fail;
    }
}

%ignore NUM_LOG_LEVELS;
%ignore FBUFSIZE;

%rename(IO) CIO;
%rename(Version) CVersion;
%rename(Parallel) CParallel;
%rename(SGObject) CSGObject;

%feature("ref")   CSGObject "SG_REF($this);"
%feature("unref") CSGObject "SG_UNREF($this);"

%include <shogun/lib/common.h>

%include "swig_typemaps.i"

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

#ifdef HAVE_BOOST_SERIALIZATION

#ifdef SWIGPYTHON
%pythoncode %{
   def __getstate__(self):
      state=self.toString()
      return state

   def __setstate__(self, state):
      self.__init__()
      self.fromString(state)

   SGObject.__setstate__=__setstate__
   SGObject.__getstate__=__getstate__
%}
#endif

#endif //HAVE_BOOST_SERIALIZATION
