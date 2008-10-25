#ifdef HAVE_PYTHON
%{
#ifndef SWIG_FILE_WITH_INIT
#  define NO_IMPORT_ARRAY
#endif
#include <stdio.h>

#include "lib/io.h"
#include "lib/common.h"
#include "lib/python.h"

/* The following code originally appeared in enthought/kiva/agg/src/numeric.i,
 * author unknown.  It was translated from C++ to C by John Hunter.  Bill
 * Spotz has modified it slightly to fix some minor bugs, add some comments
 * and some functionality.
 */

/* Macros to extract array attributes.
 */
#define is_array(a)            ((a) && PyArray_Check((PyObject *)a))
#define array_type(a)          (int)(PyArray_TYPE(a))
#define array_dimensions(a)    (((PyArrayObject *)a)->nd)
#define array_size(a,i)        (((PyArrayObject *)a)->dimensions[i])
#define array_is_contiguous(a) (PyArray_ISCONTIGUOUS(a))

/* Given a PyObject, return a string describing its type.
 */
const char* typecode_string(PyObject* py_obj) {
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable"    ;
  if (PyString_Check(  py_obj)) return "string"      ;
  if (PyInt_Check(     py_obj)) return "int"         ;
  if (PyFloat_Check(   py_obj)) return "float"       ;
  if (PyDict_Check(    py_obj)) return "dict"        ;
  if (PyList_Check(    py_obj)) return "list"        ;
  if (PyTuple_Check(   py_obj)) return "tuple"       ;
  if (PyFile_Check(    py_obj)) return "file"        ;
  if (PyModule_Check(  py_obj)) return "module"      ;
  if (PyInstance_Check(py_obj)) return "instance"    ;

  return "unkown type";
}

/* Given a numpy typecode, return a string describing the type, assuming
the following numpy type codes:

enum NPY_TYPES {    NPY_BOOL=0,
                    NPY_BYTE, NPY_UBYTE,
                    NPY_SHORT, NPY_USHORT,
                    NPY_INT, NPY_UINT,
                    NPY_LONG, NPY_ULONG,
                    NPY_LONGLONG, NPY_ULONGLONG,
                    NPY_FLOAT, NPY_DOUBLE, NPY_LONGDOUBLE,
                    NPY_CFLOAT, NPY_CDOUBLE, NPY_CLONGDOUBLE,
                    NPY_OBJECT=17,
                    NPY_STRING, NPY_UNICODE,
                    NPY_VOID,
                    NPY_NTYPES,
                    NPY_NOTYPE,
                    NPY_CHAR, 
                    NPY_USERDEF=256 
 */

const char* typecode_string(int typecode) {
  const char* type_names[24] = {"bool","byte","unsigned byte","short",
			  "unsigned short","int","unsigned int","long",
              "unsigned long","long long", "unsigned long long",
			  "float","double","long double",
              "complex float","complex double","complex long double",
			  "object","string","unicode","void","ntype","notype","char"};
  const char* user_def="user defined";

  if (typecode>24)
      return user_def;
  else
      return type_names[typecode];
}

/* Make sure input has correct numeric type.  Allow character and byte
 * to match.  Also allow int and long to match.
 */
int type_match(int actual_type, int desired_type) {
  return PyArray_EquivTypenums(actual_type, desired_type);
}

/* Given a PyObject pointer, cast it to a PyArrayObject pointer if
 * legal.  If not, set the python error string appropriately and
 * return NULL./
 */
PyArrayObject* obj_to_array_no_conversion(PyObject* input, int typecode)
{
  PyArrayObject* ary = NULL;
  if (is_array(input) && (typecode == PyArray_NOTYPE || 
			  PyArray_EquivTypenums(array_type(input), 
						typecode))) {
        ary = (PyArrayObject*) input;
    }
    else if is_array(input) {
      const char* desired_type = typecode_string(typecode);
      const char* actual_type = typecode_string(array_type(input));
      PyErr_Format(PyExc_TypeError, 
		   "Array of type '%s' required.  Array of type '%s' given", 
		   desired_type, actual_type);
      ary = NULL;
    }
    else {
      const char* desired_type = typecode_string(typecode);
      const char* actual_type = typecode_string(input);
      PyErr_Format(PyExc_TypeError, 
		   "Array of type '%s' required.  Array of type '%s' given", 
		   desired_type, actual_type);
      ary = NULL;
    }
  return ary;
}

/* Given a PyArrayObject, check to see if it is contiguous.  If so,
 * return the input pointer and flag it as not a new object.  If it is
 * not contiguous, create a new PyArrayObject using the original data,
 * flag it as a new object and return the pointer.
 * 
 * If array is NULL or dimensionality or typecode does not match
 * return NULL
 */
PyObject* make_contiguous(PyObject* ary, int* is_new_object,
                               int dims, int typecode)
{
    PyObject* array;
    if (PyArray_ISFARRAY(ary))
    {
        array = ary;
        *is_new_object = 0;
    }
    else
    {
        array=PyArray_FromAny((PyObject*)ary, NULL,0,0, NPY_FARRAY|NPY_ENSURECOPY, NULL);
        *is_new_object = 1;
    }

    if (!array)
    {
        PyErr_SetString(PyExc_TypeError, "Object did convert to Empty object - not an Array ?");
        *is_new_object=0;
        return NULL;
    }

    if (!is_array(array))
    {
        PyErr_SetString(PyExc_TypeError, "Object not an Array");
        *is_new_object=0;
        return NULL;
    }

    if (array_dimensions(array)!=dims)
    {
        PyErr_Format(PyExc_TypeError, "Array has wrong dimensionality," 
                "expected a %dd-array, received a %dd-array", dims, array_dimensions(array));
        if (*is_new_object)
            Py_DECREF(array);
        *is_new_object=0;
        return NULL;
    }

    /*this works around a numpy oddity when LONG==INT32*/
    if ((array_type(array) != typecode) &&
        !(typecode==NPY_LONG && NPY_BITSOF_INT == NPY_BITSOF_LONG 
            && NPY_BITSOF_INT==32 && array_type(array)==NPY_INT))
    {
        const char* desired_type = typecode_string(typecode);
        const char* actual_type = typecode_string(array_type(array));
        PyErr_Format(PyExc_TypeError, 
                "Array of type '%s' required.  Array of type '%s' given", 
                desired_type, actual_type);
        if (*is_new_object)
            Py_DECREF(array);
        *is_new_object=0;
        return NULL;
    }

    return array;
}

/* Test whether a python object is contiguous.  If array is
 * contiguous, return 1.  Otherwise, set the python error string and
 * return 0.
 */
int require_contiguous(PyObject* ary) {
  int contiguous = 1;
  if (!array_is_contiguous(ary)) {
    PyErr_SetString(PyExc_TypeError, "Array must be contiguous.  A discontiguous array was given");
    contiguous = 0;
  }
  return contiguous;
}

/* Require the given PyObject to have a specified number of
 * dimensions.  If the array has the specified number of dimensions,
 * return 1.  Otherwise, set the python error string and return 0.
 */
int require_dimensions(PyObject* ary, int exact_dimensions) {
  int success = 1;
  if (array_dimensions(ary) != exact_dimensions) {
    PyErr_Format(PyExc_TypeError, 
		 "Array must have %d dimensions.  Given array has %d dimensions", 
		 exact_dimensions, array_dimensions(ary));
    success = 0;
  }
  return success;
}

/* End John Hunter translation (with modifications by Bill Spotz) */
%}

%include "lib/common.i"

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int DIM1)
 *     (type* IN_ARRAY2, int DIM1, int DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the numpy.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (double* IN_ARRAY1, int DIM1) {double* series, int length}
 *     %apply (double* IN_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols}
 *     double sum(double* series, int length);
 *     double max(double* mx, int rows, int cols);
 *
 * or with
 *
 *     double sum(double* IN_ARRAY1, int DIM1);
 *     double max(double* IN_ARRAY2, int DIM1, int DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_ARRAY1, INT DIM1)
{
    $1 = (
            ($input && PyList_Check($input) && PyList_Size($input)>0) ||
            (is_array($input) && array_dimensions($input)==1 && array_type($input) == typecode)
         ) ? 1 : 0;
}

%typemap(in) (type* IN_ARRAY1, INT DIM1)
             (PyObject* array=NULL, int is_new_object)
{
    array = make_contiguous($input, &is_new_object, 1,typecode);
    if (!array)
        SWIG_fail;

    $1 = (type*) PyArray_BYTES(array);
    $2 = PyArray_DIM(array,0);
}
%typemap(freearg) (type* IN_ARRAY1, INT DIM1) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(bool,          NPY_BOOL )
TYPEMAP_IN1(char,          NPY_STRING )
TYPEMAP_IN1(uint8_t,       NPY_UINT8 )
TYPEMAP_IN1(SHORT,         NPY_INT16)
TYPEMAP_IN1(uint16_t,      NPY_UINT16 )
TYPEMAP_IN1(INT,           NPY_INT32 )
TYPEMAP_IN1(UINT,          NPY_UINT32 )
TYPEMAP_IN1(LONG,          NPY_INT64 )
TYPEMAP_IN1(ULONG,         NPY_UINT64 )
TYPEMAP_IN1(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_IN1(DREAL,         NPY_FLOAT64)
TYPEMAP_IN1(LONGREAL,      NPY_FLOAT128)
TYPEMAP_IN1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN1

 /* Two dimensional input arrays */
%define TYPEMAP_IN2(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_ARRAY2, INT DIM1, INT DIM2)
{
    $1 = (is_array($input) && array_dimensions($input)==2 &&
            array_type($input) == typecode) ? 1 : 0;
}

%typemap(in) (type* IN_ARRAY2, INT DIM1, INT DIM2)
            (PyObject* array=NULL, int is_new_object)
{
    array = make_contiguous($input, &is_new_object, 2,typecode);
    if (!array)
        SWIG_fail;

    $1 = (type*) PyArray_BYTES(array);
    $2 = PyArray_DIM(array,0);
    $3 = PyArray_DIM(array,1);
}
%typemap(freearg) (type* IN_ARRAY2, INT DIM1, INT DIM2) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN2 macros */
TYPEMAP_IN2(bool,          NPY_BOOL )
TYPEMAP_IN2(char,          NPY_STRING )
TYPEMAP_IN2(uint8_t,       NPY_UINT8 )
TYPEMAP_IN2(SHORT,         NPY_INT16)
TYPEMAP_IN2(uint16_t,      NPY_UINT16 )
TYPEMAP_IN2(INT,           NPY_INT32 )
TYPEMAP_IN2(UINT,          NPY_UINT32 )
TYPEMAP_IN2(LONG,          NPY_INT64 )
TYPEMAP_IN2(ULONG,         NPY_UINT64 )
TYPEMAP_IN2(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_IN2(DREAL,         NPY_FLOAT64)
TYPEMAP_IN2(LONGREAL,      NPY_FLOAT128)
TYPEMAP_IN2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN2


/* TYPEMAP_INPLACE macros
 *
 * This family of typemaps allows input/output C arguments of the form
 *
 *     (type* INPLACE_ARRAY1, int DIM1)
 *     (type* INPLACE_ARRAY2, int DIM1, int DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (double* INPLACE_ARRAY1, int DIM1) {double* series, int length}
 *     %apply (double* INPLACE_ARRAY2, int DIM1, int DIM2) {double* mx, int rows, int cols}
 *     void negate(double* series, int length);
 *     void normalize(double* mx, int rows, int cols);
 *     
 *
 * or with
 *
 *     void sum(double* INPLACE_ARRAY1, int DIM1);
 *     void sum(double* INPLACE_ARRAY2, int DIM1, int DIM2);
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_INPLACE1(type,typecode)
%typemap(in) (type* INPLACE_ARRAY1, INT DIM1) (PyObject* temp=NULL) {
  int i;
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp  || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = 1;
  for (i=0; i<PyArray_NDIM(temp); ++i) $2 *= PyArray_DIM(temp,i);
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE1 macro */
TYPEMAP_INPLACE1(bool,          NPY_BOOL )
TYPEMAP_INPLACE1(char,          NPY_STRING )
TYPEMAP_INPLACE1(uint8_t,       NPY_UINT8 )
TYPEMAP_INPLACE1(SHORT,         NPY_INT16)
TYPEMAP_INPLACE1(uint16_t,      NPY_UINT16 )
TYPEMAP_INPLACE1(INT,           NPY_INT32 )
TYPEMAP_INPLACE1(UINT,          NPY_UINT32 )
TYPEMAP_INPLACE1(LONG,          NPY_INT64 )
TYPEMAP_INPLACE1(ULONG,         NPY_UINT64 )
TYPEMAP_INPLACE1(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_INPLACE1(DREAL,         NPY_FLOAT64)
TYPEMAP_INPLACE1(LONGREAL,      NPY_FLOAT128)
TYPEMAP_INPLACE1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_INPLACE1

 /* Two dimensional input/output arrays */
%define TYPEMAP_INPLACE2(type,typecode)
  %typemap(in) (type* INPLACE_ARRAY2, INT DIM1, INT DIM2) (PyObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = PyArray_DIM(temp,0);
  $3 = PyArray_DIM(temp,1);
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE2 macro */
TYPEMAP_INPLACE2(bool,          NPY_BOOL )
TYPEMAP_INPLACE2(char,          NPY_STRING )
TYPEMAP_INPLACE2(uint8_t,       NPY_UINT8 )
TYPEMAP_INPLACE2(SHORT,         NPY_INT16)
TYPEMAP_INPLACE2(uint16_t,      NPY_UINT16 )
TYPEMAP_INPLACE2(INT,           NPY_INT32 )
TYPEMAP_INPLACE2(UINT,          NPY_UINT32 )
TYPEMAP_INPLACE2(LONG,          NPY_INT64 )
TYPEMAP_INPLACE2(ULONG,         NPY_UINT64 )
TYPEMAP_INPLACE2(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_INPLACE2(DREAL,         NPY_FLOAT64)
TYPEMAP_INPLACE2(LONGREAL,      NPY_FLOAT128)
TYPEMAP_INPLACE2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_INPLACE2

/* TYPEMAP_ARRAYOUT macros
 *
 * This family of typemaps allows output C arguments of the form
 *
 *     (type* ARRAYOUT_ARRAY[ANY])
 *     (type* ARRAYOUT_ARRAY[ANY][ANY])
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (double* ARRAYOUT_ARRAY[ANY] {double series, int length}
 *     %apply (double* ARRAYOUT_ARRAY[ANY][ANY]) {double* mx, int rows, int cols}
 *     void negate(double* series, int length);
 *     void normalize(double* mx, int rows, int cols);
 *     
 *
 * or with
 *
 *     void sum(double* ARRAYOUT_ARRAY[ANY]);
 *     void sum(double* ARRAYOUT_ARRAY[ANY][ANY]);
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_ARRAYOUT1(type,typecode)
%typemap(in,numinputs=0) type ARRAYOUT_ARRAY[ANY] {
  $1 = (type*) malloc($1_dim0*sizeof(type));
  if (!$1) {
    PyErr_SetString(PyExc_RuntimeError, "Failed to allocate memory");
    SWIG_fail;
  }
}
%typemap(argout) ARRAYOUT_ARRAY[ANY] {
  int dimensions[1] = {$1_dim0};
  PyObject* outArray = PyArray_FromDimsAndData(1, dimensions, typecode, (char*)$1);
}
%enddef

/* Define concrete examples of the TYPEMAP_ARRAYOUT1 macro */
TYPEMAP_ARRAYOUT1(bool,          NPY_BOOL )
TYPEMAP_ARRAYOUT1(char,          NPY_STRING )
TYPEMAP_ARRAYOUT1(uint8_t,       NPY_UINT8 )
TYPEMAP_ARRAYOUT1(SHORT,         NPY_INT16)
TYPEMAP_ARRAYOUT1(uint16_t,      NPY_UINT16 )
TYPEMAP_ARRAYOUT1(INT,           NPY_INT32 )
TYPEMAP_ARRAYOUT1(UINT,          NPY_UINT32 )
TYPEMAP_ARRAYOUT1(LONG,          NPY_INT64 )
TYPEMAP_ARRAYOUT1(ULONG,         NPY_UINT64 )
TYPEMAP_ARRAYOUT1(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_ARRAYOUT1(DREAL,         NPY_FLOAT64)
TYPEMAP_ARRAYOUT1(LONGREAL,      NPY_FLOAT128)
TYPEMAP_ARRAYOUT1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARRAYOUT1

 /* Two dimensional input/output arrays */
%define TYPEMAP_ARRAYOUT2(type,typecode)
  %typemap(in) (type* ARRAYOUT_ARRAY2, INT DIM1, INT DIM2) (PyObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = PyArray_DIM(temp,0);
  $3 = PyArray_DIM(temp,1);
}
%enddef

/* Define concrete examples of the TYPEMAP_ARRAYOUT2 macro */
TYPEMAP_ARRAYOUT2(bool,          NPY_BOOL )
TYPEMAP_ARRAYOUT2(char,          NPY_STRING )
TYPEMAP_ARRAYOUT2(uint8_t,       NPY_UINT8 )
TYPEMAP_ARRAYOUT2(SHORT,         NPY_INT16)
TYPEMAP_ARRAYOUT2(uint16_t,      NPY_UINT16 )
TYPEMAP_ARRAYOUT2(INT,           NPY_INT32 )
TYPEMAP_ARRAYOUT2(UINT,          NPY_UINT32 )
TYPEMAP_ARRAYOUT2(LONG,          NPY_INT64 )
TYPEMAP_ARRAYOUT2(ULONG,         NPY_UINT64 )
TYPEMAP_ARRAYOUT2(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_ARRAYOUT2(DREAL,         NPY_FLOAT64)
TYPEMAP_ARRAYOUT2(LONGREAL,      NPY_FLOAT128)
TYPEMAP_ARRAYOUT2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARRAYOUT2

/* TYPEMAP_ARGOUT macros
 *
 * This family of typemaps allows output C arguments of the form
 *
 *     (type** ARGOUT_ARRAY)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (DREAL** ARGOUT_ARRAY1, {(DREAL** series, INT* len)}
 *     %apply (DREAL** ARGOUT_ARRAY2, {(DREAL** matrix, INT* d1, INT* d2)}
 *
 * with
 *
 *     void sum(DREAL* series, INT* len);
 *     void sum(DREAL** series, INT* len);
 *     void sum(DREAL** matrix, INT* d1, INT* d2);
 *
 * where sum mallocs the array and assigns dimensions and the pointer
 *
 */
%define TYPEMAP_ARGOUT1(type,typecode)
%typemap(in, numinputs=0) (type** ARGOUT1, INT* DIM1) {
    $1 = (type**) malloc(sizeof(type*));
    $2 = (INT*) malloc(sizeof(INT));
}

%typemap(argout) (type** ARGOUT1, INT* DIM1) {
    npy_intp dims= (npy_intp) *$2;

    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
    if (descr && $1)
    {
        $result = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, (void*)*$1, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) $result)->flags |= NPY_OWNDATA;
    }
    else
        SWIG_fail;

    free($1); free($2);
}
%enddef

TYPEMAP_ARGOUT1(bool,          NPY_BOOL )
TYPEMAP_ARGOUT1(char,          NPY_STRING )
TYPEMAP_ARGOUT1(uint8_t,       NPY_UINT8 )
TYPEMAP_ARGOUT1(SHORT,         NPY_INT16)
TYPEMAP_ARGOUT1(uint16_t,      NPY_UINT16 )
TYPEMAP_ARGOUT1(INT,           NPY_INT32 )
TYPEMAP_ARGOUT1(UINT,          NPY_UINT32 )
TYPEMAP_ARGOUT1(LONG,          NPY_INT64 )
TYPEMAP_ARGOUT1(ULONG,         NPY_UINT64 )
TYPEMAP_ARGOUT1(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_ARGOUT1(DREAL,         NPY_FLOAT64)
TYPEMAP_ARGOUT1(LONGREAL,      NPY_FLOAT128)
TYPEMAP_ARGOUT1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARGOUT1

%define TYPEMAP_ARGOUT2(type,typecode)
%typemap(in, numinputs=0) (type** ARGOUT2, INT* DIM1, INT* DIM2) {
    $1 = (type**) malloc(sizeof(type*));
    $2 = (INT*) malloc(sizeof(INT));
    $3 = (INT*) malloc(sizeof(INT));
}

%typemap(argout) (type** ARGOUT2, INT* DIM1, INT* DIM2) {
    npy_intp dims[2]= {(npy_intp) *$2, (npy_intp) *$3};
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
    if (descr && $1)
    {
        $result=PyArray_NewFromDescr(&PyArray_Type,
                descr, 2, dims, NULL, (void*)*$1, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) $result)->flags |= NPY_OWNDATA;
    }
    else
        SWIG_fail;

    free($1); free($2); free($3);
}
%enddef

TYPEMAP_ARGOUT2(bool,          NPY_BOOL )
TYPEMAP_ARGOUT2(char,          NPY_STRING )
TYPEMAP_ARGOUT2(uint8_t,       NPY_UINT8 )
TYPEMAP_ARGOUT2(SHORT,         NPY_INT16)
TYPEMAP_ARGOUT2(uint16_t,      NPY_UINT16 )
TYPEMAP_ARGOUT2(INT,           NPY_INT32 )
TYPEMAP_ARGOUT2(UINT,          NPY_UINT32 )
TYPEMAP_ARGOUT2(LONG,          NPY_INT64 )
TYPEMAP_ARGOUT2(ULONG,         NPY_UINT64 )
TYPEMAP_ARGOUT2(SHORTREAL,     NPY_FLOAT32 )
TYPEMAP_ARGOUT2(DREAL,         NPY_FLOAT64)
TYPEMAP_ARGOUT2(LONGREAL,      NPY_FLOAT128)
TYPEMAP_ARGOUT2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARGOUT2

/* input typemap for CStringFeatures<char> */
%typemap(in) (T_STRING<char>* strings, INT num_strings, INT max_len)
{
    PyObject* list=(PyObject*) $input;
    /* Check if is a list */
    if (!list || PyList_Check(list) || PyList_Size(list)==0)
    {
        INT size=PyList_Size(list);
        T_STRING<char>* strings=new T_STRING<char>[size];

        INT max_len=0;

        for (int i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (PyString_Check(o))
            {
                INT len=PyString_Size(o);
                max_len=CMath::max(len,max_len);
                const char* str=PyString_AsString(o);

                strings[i].length=len;
                strings[i].string=NULL;

                if (len>0)
                {
                    strings[i].string=new char[len];
                    memcpy(strings[i].string, str, len);
                }
            }
            else
            {
                PyErr_SetString(PyExc_TypeError,"all elements in list must be strings");

                for (INT j=0; j<i; j++)
                    delete[] strings[i].string;
                delete[] strings;
                SWIG_fail;
            }
        }
        $1=strings;
        $2=size;
        $3=max_len;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a/empty list");
        return NULL;
    }

}
#endif
