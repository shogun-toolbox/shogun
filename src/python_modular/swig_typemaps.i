/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * This code is inspired by the python numpy.i typemaps, from John Hunter
 * and Bill Spotz that in turn is based on enthought/kiva/agg/src/numeric.i,
 * author unknown.
 *
 * It goes further by supporting strings of arbitrary types, sparse matrices
 * and ways to return arbitrariliy shaped matrices.
 *
 * Written (W) 2006-2009,2011 Soeren Sonnenburg
 * Copyright (C) 2006-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2011 Berlin Institute of Technology
 */
#ifdef HAVE_PYTHON
%{
#ifndef SWIG_FILE_WITH_INIT
#  define NO_IMPORT_ARRAY
#endif
#include <stdio.h>

#undef _POSIX_C_SOURCE
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
#include <shogun/lib/DataType.h>
}

/* Functions to extract array attributes.
 */
bool is_array(PyObject* a) { return (a) && PyArray_Check(a); }
int array_type(PyObject* a) { return (int) PyArray_TYPE(a); }
int array_dimensions(PyObject* a)  { return ((PyArrayObject *)a)->nd; }
int array_size(PyObject* a, int i) { return ((PyArrayObject *)a)->dimensions[i]; }
bool array_is_contiguous(PyObject* a) { return PyArray_ISCONTIGUOUS(a); }

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

  return "unknown type";
}

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
    else if (is_array(input)) {
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
                               int dims, int typecode, bool force_copy=false)
{
    PyObject* array;
    if (PyArray_ISFARRAY(ary) && !force_copy)
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

    if (dims!=-1 && array_dimensions(array)!=dims)
    {
        PyErr_Format(PyExc_TypeError, "Array has wrong dimensionality, " 
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


/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<type>
{
    $1 = (
            ($input && PyList_Check($input) && PyList_Size($input)>0) ||
            (is_array($input) && array_dimensions($input)==1 && array_type($input) == typecode)
         ) ? 1 : 0;
}

%typemap(in) shogun::SGVector<type>
{
    int is_new_object;
    PyObject* array = make_contiguous($input, &is_new_object, 1,typecode, true);
    if (!array)
        SWIG_fail;

    $1 = shogun::SGVector<type>((type*) PyArray_BYTES(array), PyArray_DIM(array,0));
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGVECTOR macros */
TYPEMAP_IN_SGVECTOR(bool,          NPY_BOOL)
TYPEMAP_IN_SGVECTOR(char,          NPY_STRING)
TYPEMAP_IN_SGVECTOR(uint8_t,       NPY_UINT8)
TYPEMAP_IN_SGVECTOR(int16_t,       NPY_INT16)
TYPEMAP_IN_SGVECTOR(uint16_t,      NPY_UINT16)
TYPEMAP_IN_SGVECTOR(int32_t,       NPY_INT32)
TYPEMAP_IN_SGVECTOR(uint32_t,      NPY_UINT32)
TYPEMAP_IN_SGVECTOR(int64_t,       NPY_INT64)
TYPEMAP_IN_SGVECTOR(uint64_t,      NPY_UINT64)
TYPEMAP_IN_SGVECTOR(float32_t,     NPY_FLOAT32)
TYPEMAP_IN_SGVECTOR(float64_t,     NPY_FLOAT64)
TYPEMAP_IN_SGVECTOR(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_IN_SGVECTOR(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN_SGVECTOR

/* One dimensional input arrays */
%define TYPEMAP_OUT_SGVECTOR(type,typecode)
%typemap(out) shogun::SGVector<type>
{
    npy_intp dims= (npy_intp) $1.length;
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
    if (descr)
    {
        $result = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, (void*) $1.vector, NPY_FARRAY | NPY_WRITEABLE, NULL);
        /*((PyArrayObject*) $result)->flags |= NPY_OWNDATA;*/
    }
    else
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGVECTOR macros */
TYPEMAP_OUT_SGVECTOR(bool,          NPY_BOOL)
TYPEMAP_OUT_SGVECTOR(char,          NPY_STRING)
TYPEMAP_OUT_SGVECTOR(uint8_t,       NPY_UINT8)
TYPEMAP_OUT_SGVECTOR(int16_t,       NPY_INT16)
TYPEMAP_OUT_SGVECTOR(uint16_t,      NPY_UINT16)
TYPEMAP_OUT_SGVECTOR(int32_t,       NPY_INT32)
TYPEMAP_OUT_SGVECTOR(uint32_t,      NPY_UINT32)
TYPEMAP_OUT_SGVECTOR(int64_t,       NPY_INT64)
TYPEMAP_OUT_SGVECTOR(uint64_t,      NPY_UINT64)
TYPEMAP_OUT_SGVECTOR(float32_t,     NPY_FLOAT32)
TYPEMAP_OUT_SGVECTOR(float64_t,     NPY_FLOAT64)
TYPEMAP_OUT_SGVECTOR(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_OUT_SGVECTOR(PyObject,      NPY_OBJECT)

#undef TYPEMAP_OUT_SGVECTOR

/* Two dimensional input arrays */
%define TYPEMAP_IN_SGMATRIX(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGMatrix<type>
{
    $1 = (is_array($input) && array_dimensions($input)==2 &&
            array_type($input) == typecode) ? 1 : 0;
}

%typemap(in) shogun::SGMatrix<type>
{
    int is_new_object;
    PyObject* array = make_contiguous($input, &is_new_object, 2,typecode, true);
    if (!array)
        SWIG_fail;

    $1 = shogun::SGMatrix<type>((type*) PyArray_BYTES(array),
            PyArray_DIM(array,0), PyArray_DIM(array,1));
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGMATRIX macros */
TYPEMAP_IN_SGMATRIX(bool,          NPY_BOOL)
TYPEMAP_IN_SGMATRIX(char,          NPY_STRING)
TYPEMAP_IN_SGMATRIX(uint8_t,       NPY_UINT8)
TYPEMAP_IN_SGMATRIX(int16_t,       NPY_INT16)
TYPEMAP_IN_SGMATRIX(uint16_t,      NPY_UINT16)
TYPEMAP_IN_SGMATRIX(int32_t,       NPY_INT32)
TYPEMAP_IN_SGMATRIX(uint32_t,      NPY_UINT32)
TYPEMAP_IN_SGMATRIX(int64_t,       NPY_INT64)
TYPEMAP_IN_SGMATRIX(uint64_t,      NPY_UINT64)
TYPEMAP_IN_SGMATRIX(float32_t,     NPY_FLOAT32)
TYPEMAP_IN_SGMATRIX(float64_t,     NPY_FLOAT64)
TYPEMAP_IN_SGMATRIX(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_IN_SGMATRIX(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN_SGMATRIX

/* One dimensional input arrays */
%define TYPEMAP_OUT_SGMATRIX(type,typecode)
%typemap(out) shogun::SGMatrix<type>
{
    npy_intp dims[2]= {(npy_intp) $1.num_rows, (npy_intp) $1.num_cols };
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);
    if (descr)
    {
        $result = PyArray_NewFromDescr(&PyArray_Type,
                descr, 2, dims, NULL, (void*) $1.matrix, NPY_FARRAY | NPY_WRITEABLE, NULL);
        /*((PyArrayObject*) $result)->flags |= NPY_OWNDATA;*/
    }
    else
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGMATRIX macros */
TYPEMAP_OUT_SGMATRIX(bool,          NPY_BOOL)
TYPEMAP_OUT_SGMATRIX(char,          NPY_STRING)
TYPEMAP_OUT_SGMATRIX(uint8_t,       NPY_UINT8)
TYPEMAP_OUT_SGMATRIX(int16_t,       NPY_INT16)
TYPEMAP_OUT_SGMATRIX(uint16_t,      NPY_UINT16)
TYPEMAP_OUT_SGMATRIX(int32_t,       NPY_INT32)
TYPEMAP_OUT_SGMATRIX(uint32_t,      NPY_UINT32)
TYPEMAP_OUT_SGMATRIX(int64_t,       NPY_INT64)
TYPEMAP_OUT_SGMATRIX(uint64_t,      NPY_UINT64)
TYPEMAP_OUT_SGMATRIX(float32_t,     NPY_FLOAT32)
TYPEMAP_OUT_SGMATRIX(float64_t,     NPY_FLOAT64)
TYPEMAP_OUT_SGMATRIX(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_OUT_SGMATRIX(PyObject,      NPY_OBJECT)

#undef TYPEMAP_OUT_SGMATRIX

/* N-dimensional input arrays */
%define TYPEMAP_INND(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        shogun::SGNDArray<type>
{
    $1 = (is_array($input)) ? 1 : 0;
}

%typemap(in) shogun::SGNDArray<type>
{
    (PyObject* array=NULL, int is_new_object, int32_t* temp_dims=NULL)
    array = make_contiguous($input, &is_new_object, -1,typecode);
    if (!array)
        SWIG_fail;

    int32_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      SWIG_fail;

    temp_dims = new int32_t[ndim];

    npy_intp* py_dims = PyArray_DIMS(array);

    for (int32_t i=0; i<ndim; i++)
      temp_dims[i] = py_dims[i];
    
    $1 = SGNDArray((type*) PyArray_BYTES(array), temp_dims, ndim)
}
%typemap(freearg) shogun::SGNDArray<type>
{
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
  delete[] temp_dims$argnum;
}
%enddef

/* Define concrete examples of the TYPEMAP_INND macros */
TYPEMAP_INND(bool,          NPY_BOOL)
TYPEMAP_INND(char,          NPY_STRING)
TYPEMAP_INND(uint8_t,       NPY_UINT8)
TYPEMAP_INND(int16_t,       NPY_INT16)
TYPEMAP_INND(uint16_t,      NPY_UINT16)
TYPEMAP_INND(int32_t,       NPY_INT32)
TYPEMAP_INND(uint32_t,      NPY_UINT32)
TYPEMAP_INND(int64_t,       NPY_INT64)
TYPEMAP_INND(uint64_t,      NPY_UINT64)
TYPEMAP_INND(float32_t,     NPY_FLOAT32)
TYPEMAP_INND(float64_t,     NPY_FLOAT64)
TYPEMAP_INND(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_INND(PyObject,      NPY_OBJECT)

#undef TYPEMAP_INND

/* input typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGStringList<type>
{
    PyObject* list=(PyObject*) $input;

    $1=0;
    if (list && PyList_Check(list) && PyList_Size(list)>0)
    {
        $1=1;
        int32_t size=PyList_Size(list);
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (typecode == NPY_STRING)
            {
                if (!PyString_Check(o))
                {
                    $1=0;
                    break;
                }
            }
            else
            {
                if (!is_array(o) || array_dimensions(o)!=1 || array_type(o) != typecode)
                {
                    $1=0;
                    break;
                }
            }
        }
    }
}
%typemap(in) shogun::SGStringList<type>
{
    PyObject* list=(PyObject*) $input;
    /* Check if is a list */
    if (!list || PyList_Check(list) || PyList_Size(list)==0)
    {
        int32_t size=PyList_Size(list);
        shogun::SGString<type>* strings=new shogun::SGString<type>[size];

        int32_t max_len=0;
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (typecode == NPY_STRING)
            {
                if (PyString_Check(o))
                {
                    int32_t len=PyString_Size(o);
                    max_len=shogun::CMath::max(len,max_len);
                    const char* str=PyString_AsString(o);

                    strings[i].length=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=new type[len];
                        memcpy(strings[i].string, str, len);
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be strings");

                    for (int32_t j=0; j<i; j++)
                        delete[] strings[i].string;
                    delete[] strings;
                    SWIG_fail;
                }
            }
            else
            {
                if (is_array(o) && array_dimensions(o)==1 && array_type(o) == typecode)
                {
                    int is_new_object=0;
                    PyObject* array = make_contiguous(o, &is_new_object, 1, typecode);
                    if (!array)
                        SWIG_fail;

                    type* str=(type*) PyArray_BYTES(array);
                    int32_t len = PyArray_DIM(array,0);
                    max_len=shogun::CMath::max(len,max_len);

                    strings[i].length=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=new type[len];
                        memcpy(strings[i].string, str, len*sizeof(type));
                    }

                    if (is_new_object)
                        Py_DECREF(array);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be of same array type");

                    for (int32_t j=0; j<i; j++)
                        delete[] strings[i].string;
                    delete[] strings;
                    SWIG_fail;
                }
            }
        }
        SGStringList<type> sl;
        sl.strings=strings;
        sl.num_strings=size;
        sl.max_string_length=max_len;
        $1=sl;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a/empty list");
        return NULL;
    }
}
%enddef

TYPEMAP_STRINGFEATURES_IN(bool,          NPY_BOOL)
TYPEMAP_STRINGFEATURES_IN(char,          NPY_STRING)
TYPEMAP_STRINGFEATURES_IN(uint8_t,       NPY_UINT8)
TYPEMAP_STRINGFEATURES_IN(int16_t,       NPY_INT16)
TYPEMAP_STRINGFEATURES_IN(uint16_t,      NPY_UINT16)
TYPEMAP_STRINGFEATURES_IN(int32_t,       NPY_INT32)
TYPEMAP_STRINGFEATURES_IN(uint32_t,      NPY_UINT32)
TYPEMAP_STRINGFEATURES_IN(int64_t,       NPY_INT64)
TYPEMAP_STRINGFEATURES_IN(uint64_t,      NPY_UINT64)
TYPEMAP_STRINGFEATURES_IN(float32_t,     NPY_FLOAT32)
TYPEMAP_STRINGFEATURES_IN(float64_t,     NPY_FLOAT64)
TYPEMAP_STRINGFEATURES_IN(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_IN(PyObject,      NPY_OBJECT)

#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGString<type>
{
    shogun::SGString<type>* str=$1.strings;
    int32_t num=$1.num_strings;
    PyObject* list = PyList_New(num);

    if (list && str)
    {
        for (int32_t i=0; i<num; i++)
        {
            PyObject* s=NULL;

            if (typecode == NPY_STRING)
            {
                /* This path is only taking if str[i].string is a char*. However this cast is
                   required to build through for non char types. */
                s=PyString_FromStringAndSize((char*) str[i].string, str[i].length);
            }
            else
            {
                PyArray_Descr* descr=PyArray_DescrFromType(typecode);
                type* data = (type*) malloc(str[i].length*sizeof(type));
                if (descr && data)
                {
                    memcpy(data, str[i].string, str[i].length*sizeof(type));
                    npy_intp dims = str[i].length;

                    s = PyArray_NewFromDescr(&PyArray_Type,
                            descr, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
                    ((PyArrayObject*) s)->flags |= NPY_OWNDATA;
                }
                else
                    SWIG_fail;
            }

            PyList_SetItem(list, i, s);
            delete[] str[i].string;
        }
        delete[] str;
        $result = list;
    }
    else
        SWIG_fail;
}
%enddef

TYPEMAP_STRINGFEATURES_OUT(bool,          NPY_BOOL)
TYPEMAP_STRINGFEATURES_OUT(char,          NPY_STRING)
TYPEMAP_STRINGFEATURES_OUT(uint8_t,       NPY_UINT8)
TYPEMAP_STRINGFEATURES_OUT(int16_t,       NPY_INT16)
TYPEMAP_STRINGFEATURES_OUT(uint16_t,      NPY_UINT16)
TYPEMAP_STRINGFEATURES_OUT(int32_t,       NPY_INT32)
TYPEMAP_STRINGFEATURES_OUT(uint32_t,      NPY_UINT32)
TYPEMAP_STRINGFEATURES_OUT(int64_t,       NPY_INT64)
TYPEMAP_STRINGFEATURES_OUT(uint64_t,      NPY_UINT64)
TYPEMAP_STRINGFEATURES_OUT(float32_t,     NPY_FLOAT32)
TYPEMAP_STRINGFEATURES_OUT(float64_t,     NPY_FLOAT64)
TYPEMAP_STRINGFEATURES_OUT(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_OUT(PyObject,      NPY_OBJECT)
#undef TYPEMAP_STRINGFEATURES_ARGOUT


/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGSparseMatrix<type>
{
    $1 = ( PyObject_HasAttrString($input, "indptr") &&
            PyObject_HasAttrString($input, "indices") &&
            PyObject_HasAttrString($input, "data") &&
            PyObject_HasAttrString($input, "shape")
         ) ? 1 : 0;
}

%typemap(in) shogun::SGSparseMatrix<type>
{
    PyObject* o=(PyObject*) $input;

    /* a column compressed storage sparse matrix in python scipy
       looks like this

       A = csc_matrix( ... )
       A.indptr # pointer array
       A.indices # indices array
       A.data # nonzero values array
       A.shape # size of matrix

       >>> type(A.indptr)
       <type 'numpy.ndarray'> #int32
       >>> type(A.indices)
       <type 'numpy.ndarray'> #int32
       >>> type(A.data)
       <type 'numpy.ndarray'>
       >>> type(A.shape)
       <type 'tuple'>
     */

    if ( PyObject_HasAttrString(o, "indptr") &&
            PyObject_HasAttrString(o, "indices") &&
            PyObject_HasAttrString(o, "data") &&
            PyObject_HasAttrString(o, "shape"))
    {
        /* fetch sparse attributes */
        PyObject* indptr = PyObject_GetAttrString(o, "indptr");
        PyObject* indices = PyObject_GetAttrString(o, "indices");
        PyObject* data = PyObject_GetAttrString(o, "data");
        PyObject* shape = PyObject_GetAttrString(o, "shape");

        /* check that types are OK */
        if ((!is_array(indptr)) || (array_dimensions(indptr)!=1) ||
                (array_type(indptr)!=NPY_INT && array_type(indptr)!=NPY_LONG))
        {
            PyErr_SetString(PyExc_TypeError,"indptr array should be 1d int's");
            return NULL;
        }

        if (!is_array(indices) || array_dimensions(indices)!=1 ||
                (array_type(indices)!=NPY_INT && array_type(indices)!=NPY_LONG))
        {
            PyErr_SetString(PyExc_TypeError,"indices array should be 1d int's");
            return NULL;
        }

        if (!is_array(data) || array_dimensions(data)!=1 || array_type(data) != typecode)
        {
            PyErr_SetString(PyExc_TypeError,"data array should be 1d and match datatype");
            return NULL;
        }

        if (!PyTuple_Check(shape))
        {
            PyErr_SetString(PyExc_TypeError,"shape should be a tuple");
            return NULL;
        }

        /* get array dimensions */
        int32_t num_feat=PyInt_AsLong(PyTuple_GetItem(shape, 0));
        int32_t num_vec=PyInt_AsLong(PyTuple_GetItem(shape, 1));

        /* get indptr array */
        int is_new_object_indptr=0;
        PyObject* array_indptr = make_contiguous(indptr, &is_new_object_indptr, 1, NPY_INT32);
        if (!array_indptr) SWIG_fail;
        int32_t* bytes_indptr=(int32_t*) PyArray_BYTES(array_indptr);
        int32_t len_indptr = PyArray_DIM(array_indptr,0);

        /* get indices array */
        int is_new_object_indices=0;
        PyObject* array_indices = make_contiguous(indices, &is_new_object_indices, 1, NPY_INT32);
        if (!array_indices) SWIG_fail;
        int32_t* bytes_indices=(int32_t*) PyArray_BYTES(array_indices);
        int32_t len_indices = PyArray_DIM(array_indices,0);

        /* get data array */
        int is_new_object_data=0;
        PyObject* array_data = make_contiguous(data, &is_new_object_data, 1, typecode);
        if (!array_data) SWIG_fail;
        type* bytes_data=(type*) PyArray_BYTES(array_data);
        int32_t len_data = PyArray_DIM(array_data,0);

        if (len_indices!=len_data)
            SWIG_fail;

        shogun::SGSparseVector<type>* sfm = new shogun::SGSparseVector<type>[num_vec];

        for (int32_t i=0; i<num_vec; i++)
        {
            sfm[i].vec_index = i;
            sfm[i].num_feat_entries = 0;
            sfm[i].features = NULL;
        }

        for (int32_t i=1; i<len_indptr; i++)
        {
            int32_t num = bytes_indptr[i]-bytes_indptr[i-1];
            
            if (num>0)
            {
                shogun::SGSparseVectorEntry<type>* features=new shogun::SGSparseVectorEntry<type>[num];

                for (int32_t j=0; j<num; j++)
                {
                    features[j].feat_index=*bytes_indices;
                    features[j].entry=*bytes_data;

                    bytes_indices++;
                    bytes_data++;
                }
                sfm[i-1].num_feat_entries=num;
                sfm[i-1].features=features;
            }
        }

        if (is_new_object_indptr)
            Py_DECREF(array_indptr);
        if (is_new_object_indices)
            Py_DECREF(array_indices);
        if (is_new_object_data)
            Py_DECREF(array_data);

        Py_DECREF(indptr);
        Py_DECREF(indices);
        Py_DECREF(data);
        Py_DECREF(shape);

        SGSparseMatrix<type> sm;
        sm.sparse_matrix=sfm;
        sm.num_features=num_feat;
        sm.num_vectors=num_vec;
        $1=sm;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a column compressed sparse matrix");
        return NULL;
    }
}
%enddef

TYPEMAP_SPARSEFEATURES_IN(bool,          NPY_BOOL)
TYPEMAP_SPARSEFEATURES_IN(char,          NPY_STRING)
TYPEMAP_SPARSEFEATURES_IN(uint8_t,       NPY_UINT8)
TYPEMAP_SPARSEFEATURES_IN(int16_t,       NPY_INT16)
TYPEMAP_SPARSEFEATURES_IN(uint16_t,      NPY_UINT16)
TYPEMAP_SPARSEFEATURES_IN(int32_t,       NPY_INT32)
TYPEMAP_SPARSEFEATURES_IN(uint32_t,      NPY_UINT32)
TYPEMAP_SPARSEFEATURES_IN(int64_t,       NPY_INT64)
TYPEMAP_SPARSEFEATURES_IN(uint64_t,      NPY_UINT64)
TYPEMAP_SPARSEFEATURES_IN(float32_t,     NPY_FLOAT32)
TYPEMAP_SPARSEFEATURES_IN(float64_t,     NPY_FLOAT64)
TYPEMAP_SPARSEFEATURES_IN(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_IN(PyObject,      NPY_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_OUT(type,typecode)
%typemap(out) shogun::SGSparseVector<type>
    
{
    shogun::SGSparseVector<type>* sfm=$1.sparse_matrix;
    int32_t num_feat=$1.num_features;
    int32_t num_vec=$1.num_vectors;

    int64_t nnz=0;
    for (int32_t i=0; i<num_vec; i++)
        nnz+=sfm[i].num_feat_entries;

    PyObject* tuple = PyTuple_New(3);

    if (tuple && sfm)
    {
        PyObject* data_py=NULL;
        PyObject* indices_py=NULL;
        PyObject* indptr_py=NULL;

        PyArray_Descr* descr=PyArray_DescrFromType(NPY_INT32);
        PyArray_Descr* descr_data=PyArray_DescrFromType(typecode);

        int32_t* indptr = (int32_t*) malloc((num_vec+1)*sizeof(int32_t));
        int32_t* indices = (int32_t*) malloc(nnz*sizeof(int32_t));
        type* data = (type*) malloc(nnz*sizeof(type));

        if (descr && descr_data && indptr && indices && data)
        {
            indptr[0]=0;

            int32_t* i_ptr=indices;
            type* d_ptr=data;

            for (int32_t i=0; i<num_vec; i++)
            {
                indptr[i+1]=indptr[i];
                if (sfm[i].vec_index==i)
                {
                    indptr[i+1]+=sfm[i].num_feat_entries;

                    for (int32_t j=0; j<sfm[i].num_feat_entries; j++)
                    {
                        *i_ptr=sfm[i].features[j].feat_index;
                        *d_ptr=sfm[i].features[j].entry;

                        i_ptr++;
                        d_ptr++;
                    }
                }
            }

            npy_intp indptr_dims = num_vec+1;
            indptr_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &indptr_dims, NULL, (void*) indptr, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) indptr_py)->flags |= NPY_OWNDATA;

            npy_intp dims = nnz;
            indices_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &dims, NULL, (void*) indices, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) indices_py)->flags |= NPY_OWNDATA;

            data_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr_data, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) data_py)->flags |= NPY_OWNDATA;

            PyTuple_SetItem(tuple, 0, data_py);
            PyTuple_SetItem(tuple, 1, indices_py);
            PyTuple_SetItem(tuple, 2, indptr_py);
            $result = tuple;
        }
        else
            SWIG_fail;
    }
    else
        SWIG_fail;
}
%enddef

TYPEMAP_SPARSEFEATURES_OUT(bool,          NPY_BOOL)
TYPEMAP_SPARSEFEATURES_OUT(char,          NPY_STRING)
TYPEMAP_SPARSEFEATURES_OUT(uint8_t,       NPY_UINT8)
TYPEMAP_SPARSEFEATURES_OUT(int16_t,       NPY_INT16)
TYPEMAP_SPARSEFEATURES_OUT(uint16_t,      NPY_UINT16)
TYPEMAP_SPARSEFEATURES_OUT(int32_t,       NPY_INT32)
TYPEMAP_SPARSEFEATURES_OUT(uint32_t,      NPY_UINT32)
TYPEMAP_SPARSEFEATURES_OUT(int64_t,       NPY_INT64)
TYPEMAP_SPARSEFEATURES_OUT(uint64_t,      NPY_UINT64)
TYPEMAP_SPARSEFEATURES_OUT(float32_t,     NPY_FLOAT32)
TYPEMAP_SPARSEFEATURES_OUT(float64_t,     NPY_FLOAT64)
TYPEMAP_SPARSEFEATURES_OUT(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_OUT(PyObject,      NPY_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_OUT












/* OBSOLETE TYPEMAPS FOLLOW */

/* TYPEMAP_IN macros
 *
 * This family of typemaps allows pure input C arguments of the form
 *
 *     (type* IN_ARRAY1, int32_t DIM1)
 *     (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single array (or any
 * python object that can be passed to the numpy.array constructor
 * to produce an arrayof te specified shape).  This can be applied to
 * a existing functions using the %apply directive:
 *
 *     %apply (float64_t* IN_ARRAY1, int32_t DIM1) {float64_t* series, int32_t length}
 *     %apply (float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {float64_t* mx, int32_t rows, int32_t cols}
 *     float64_t sum(float64_t* series, int32_t length);
 *     float64_t max(float64_t* mx, int32_t rows, int32_t cols);
 *
 * or with
 *
 *     float64_t sum(float64_t* IN_ARRAY1, int32_t DIM1);
 *     float64_t max(float64_t* IN_ARRAY2, int32_t DIM1, int32_t DIM2);
 */

/* One dimensional input arrays */
%define TYPEMAP_IN1(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_ARRAY1, int32_t DIM1)
{
    $1 = (
            ($input && PyList_Check($input) && PyList_Size($input)>0) ||
            (is_array($input) && array_dimensions($input)==1 && array_type($input) == typecode)
         ) ? 1 : 0;
}

%typemap(in) (type* IN_ARRAY1, int32_t DIM1)
             (PyObject* array=NULL, int is_new_object)
{
    array = make_contiguous($input, &is_new_object, 1,typecode);
    if (!array)
        SWIG_fail;

    $1 = (type*) PyArray_BYTES(array);
    $2 = PyArray_DIM(array,0);
}
%typemap(freearg) (type* IN_ARRAY1, int32_t DIM1) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(bool,          NPY_BOOL)
TYPEMAP_IN1(char,          NPY_STRING)
TYPEMAP_IN1(uint8_t,       NPY_UINT8)
TYPEMAP_IN1(int16_t,       NPY_INT16)
TYPEMAP_IN1(uint16_t,      NPY_UINT16)
TYPEMAP_IN1(int32_t,       NPY_INT32)
TYPEMAP_IN1(uint32_t,      NPY_UINT32)
TYPEMAP_IN1(int64_t,       NPY_INT64)
TYPEMAP_IN1(uint64_t,      NPY_UINT64)
TYPEMAP_IN1(float32_t,     NPY_FLOAT32)
TYPEMAP_IN1(float64_t,     NPY_FLOAT64)
TYPEMAP_IN1(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_IN1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN1

/* One dimensional input arrays */
%define TYPEMAP_IN1(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_ARRAY1, int64_t DIM1)
{
    $1 = (
            ($input && PyList_Check($input) && PyList_Size($input)>0) ||
            (is_array($input) && array_dimensions($input)==1 && array_type($input) == typecode)
         ) ? 1 : 0;
}

%typemap(in) (type* IN_ARRAY1, int64_t DIM1)
             (PyObject* array=NULL, int is_new_object)
{
    array = make_contiguous($input, &is_new_object, 1,typecode);
    if (!array)
        SWIG_fail;

    $1 = (type*) PyArray_BYTES(array);
    $2 = PyArray_DIM(array,0);
}
%typemap(freearg) (type* IN_ARRAY1, int64_t DIM1) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN1 macros */
TYPEMAP_IN1(bool,          NPY_BOOL)
TYPEMAP_IN1(char,          NPY_STRING)
TYPEMAP_IN1(uint8_t,       NPY_UINT8)
TYPEMAP_IN1(int16_t,       NPY_INT16)
TYPEMAP_IN1(uint16_t,      NPY_UINT16)
TYPEMAP_IN1(int32_t,       NPY_INT32)
TYPEMAP_IN1(uint32_t,      NPY_UINT32)
TYPEMAP_IN1(int64_t,       NPY_INT64)
TYPEMAP_IN1(uint64_t,      NPY_UINT64)
TYPEMAP_IN1(float32_t,     NPY_FLOAT32)
TYPEMAP_IN1(float64_t,     NPY_FLOAT64)
TYPEMAP_IN1(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_IN1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN1


 /* Two dimensional input arrays */
%define TYPEMAP_IN2(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
{
    $1 = (is_array($input) && array_dimensions($input)==2 &&
            array_type($input) == typecode) ? 1 : 0;
}

%typemap(in) (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2)
            (PyObject* array=NULL, int is_new_object)
{
    array = make_contiguous($input, &is_new_object, 2,typecode);
    if (!array)
        SWIG_fail;

    $1 = (type*) PyArray_BYTES(array);
    $2 = PyArray_DIM(array,0);
    $3 = PyArray_DIM(array,1);
}
%typemap(freearg) (type* IN_ARRAY2, int32_t DIM1, int32_t DIM2) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
}
%enddef

/* Define concrete examples of the TYPEMAP_IN2 macros */
TYPEMAP_IN2(bool,          NPY_BOOL)
TYPEMAP_IN2(char,          NPY_STRING)
TYPEMAP_IN2(uint8_t,       NPY_UINT8)
TYPEMAP_IN2(int16_t,       NPY_INT16)
TYPEMAP_IN2(uint16_t,      NPY_UINT16)
TYPEMAP_IN2(int32_t,       NPY_INT32)
TYPEMAP_IN2(uint32_t,      NPY_UINT32)
TYPEMAP_IN2(int64_t,       NPY_INT64)
TYPEMAP_IN2(uint64_t,      NPY_UINT64)
TYPEMAP_IN2(float32_t,     NPY_FLOAT32)
TYPEMAP_IN2(float64_t,     NPY_FLOAT64)
TYPEMAP_IN2(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_IN2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_IN2

/* N-dimensional input arrays */
%define TYPEMAP_INND(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (type* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS)
{
    $1 = (is_array($input)) ? 1 : 0;
}

%typemap(in) (type* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS)
            (PyObject* array=NULL, int is_new_object, int32_t* temp_dims=NULL)
{
    array = make_contiguous($input, &is_new_object, -1,typecode);
    if (!array)
        SWIG_fail;

    int32_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      SWIG_fail;

    temp_dims = new int32_t[ndim];

    npy_intp* py_dims = PyArray_DIMS(array);

    for (int32_t i=0; i<ndim; i++)
      temp_dims[i] = py_dims[i];
    
    $1 = (type*) PyArray_BYTES(array);
    $2 = temp_dims;
    $3 = ndim;
}
%typemap(freearg) (type* IN_NDARRAY, int32_t* DIMS, int32_t NDIMS) {
  if (is_new_object$argnum && array$argnum) Py_DECREF(array$argnum);
  delete[] temp_dims$argnum;
}
%enddef

/* Define concrete examples of the TYPEMAP_INND macros */
TYPEMAP_INND(bool,          NPY_BOOL)
TYPEMAP_INND(char,          NPY_STRING)
TYPEMAP_INND(uint8_t,       NPY_UINT8)
TYPEMAP_INND(int16_t,       NPY_INT16)
TYPEMAP_INND(uint16_t,      NPY_UINT16)
TYPEMAP_INND(int32_t,       NPY_INT32)
TYPEMAP_INND(uint32_t,      NPY_UINT32)
TYPEMAP_INND(int64_t,       NPY_INT64)
TYPEMAP_INND(uint64_t,      NPY_UINT64)
TYPEMAP_INND(float32_t,     NPY_FLOAT32)
TYPEMAP_INND(float64_t,     NPY_FLOAT64)
TYPEMAP_INND(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_INND(PyObject,      NPY_OBJECT)

#undef TYPEMAP_INND


/* TYPEMAP_INPLACE macros
 *
 * This family of typemaps allows input/output C arguments of the form
 *
 *     (type* INPLACE_ARRAY1, int32_t DIM1)
 *     (type* INPLACE_ARRAY2, int32_t DIM1, int32_t DIM2)
 *
 * where "type" is any type supported by the numpy module, to be
 * called in python with an argument list of a single contiguous
 * numpy array.  This can be applied to an existing function using
 * the %apply directive:
 *
 *     %apply (float64_t* INPLACE_ARRAY1, int32_t DIM1) {float64_t* series, int32_t length}
 *     %apply (float64_t* INPLACE_ARRAY2, int32_t DIM1, int32_t DIM2) {float64_t* mx, int32_t rows, int32_t cols}
 *     void negate(float64_t* series, int32_t length);
 *     void normalize(float64_t* mx, int32_t rows, int32_t cols);
 *     
 *
 * or with
 *
 *     void sum(float64_t* INPLACE_ARRAY1, int32_t DIM1);
 *     void sum(float64_t* INPLACE_ARRAY2, int32_t DIM1, int32_t DIM2);
 */

 /* One dimensional input/output arrays */
%define TYPEMAP_INPLACE1(type,typecode)
%typemap(in) (type* INPLACE_ARRAY1, int32_t DIM1) (PyObject* temp=NULL) {
  int i;
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp  || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = 1;
  for (i=0; i<PyArray_NDIM(temp); ++i) $2 *= PyArray_DIM(temp,i);
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE1 macro */
TYPEMAP_INPLACE1(bool,          NPY_BOOL)
TYPEMAP_INPLACE1(char,          NPY_STRING)
TYPEMAP_INPLACE1(uint8_t,       NPY_UINT8)
TYPEMAP_INPLACE1(int16_t,       NPY_INT16)
TYPEMAP_INPLACE1(uint16_t,      NPY_UINT16)
TYPEMAP_INPLACE1(int32_t,       NPY_INT32)
TYPEMAP_INPLACE1(uint32_t,      NPY_UINT32)
TYPEMAP_INPLACE1(int64_t,       NPY_INT64)
TYPEMAP_INPLACE1(uint64_t,      NPY_UINT64)
TYPEMAP_INPLACE1(float32_t,     NPY_FLOAT32)
TYPEMAP_INPLACE1(float64_t,     NPY_FLOAT64)
TYPEMAP_INPLACE1(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_INPLACE1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_INPLACE1

 /* Two dimensional input/output arrays */
%define TYPEMAP_INPLACE2(type,typecode)
  %typemap(in) (type* INPLACE_ARRAY2, int32_t DIM1, int32_t DIM2) (PyObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = PyArray_DIM(temp,0);
  $3 = PyArray_DIM(temp,1);
}
%enddef

/* Define concrete examples of the TYPEMAP_INPLACE2 macro */
TYPEMAP_INPLACE2(bool,          NPY_BOOL)
TYPEMAP_INPLACE2(char,          NPY_STRING)
TYPEMAP_INPLACE2(uint8_t,       NPY_UINT8)
TYPEMAP_INPLACE2(int16_t,       NPY_INT16)
TYPEMAP_INPLACE2(uint16_t,      NPY_UINT16)
TYPEMAP_INPLACE2(int32_t,       NPY_INT32)
TYPEMAP_INPLACE2(uint32_t,      NPY_UINT32)
TYPEMAP_INPLACE2(int64_t,       NPY_INT64)
TYPEMAP_INPLACE2(uint64_t,      NPY_UINT64)
TYPEMAP_INPLACE2(float32_t,     NPY_FLOAT32)
TYPEMAP_INPLACE2(float64_t,     NPY_FLOAT64)
TYPEMAP_INPLACE2(floatmax_t,    NPY_LONGDOUBLE)
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
 *     %apply (float64_t* ARRAYOUT_ARRAY[ANY] {float64_t series, int32_t length}
 *     %apply (float64_t* ARRAYOUT_ARRAY[ANY][ANY]) {float64_t* mx, int32_t rows, int32_t cols}
 *     void negate(float64_t* series, int32_t length);
 *     void normalize(float64_t* mx, int32_t rows, int32_t cols);
 *     
 *
 * or with
 *
 *     void sum(float64_t* ARRAYOUT_ARRAY[ANY]);
 *     void sum(float64_t* ARRAYOUT_ARRAY[ANY][ANY]);
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
    npy_intp dims = $1_dim0;
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr && $1)
    {
        $result = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, (void*)*$1, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) $result)->flags |= NPY_OWNDATA;
    }
    else
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_ARRAYOUT1 macro */
TYPEMAP_ARRAYOUT1(bool,          NPY_BOOL)
TYPEMAP_ARRAYOUT1(char,          NPY_STRING)
TYPEMAP_ARRAYOUT1(uint8_t,       NPY_UINT8)
TYPEMAP_ARRAYOUT1(int16_t,       NPY_INT16)
TYPEMAP_ARRAYOUT1(uint16_t,      NPY_UINT16)
TYPEMAP_ARRAYOUT1(int32_t,       NPY_INT32)
TYPEMAP_ARRAYOUT1(uint32_t,      NPY_UINT32)
TYPEMAP_ARRAYOUT1(int64_t,       NPY_INT64)
TYPEMAP_ARRAYOUT1(uint64_t,      NPY_UINT64)
TYPEMAP_ARRAYOUT1(float32_t,     NPY_FLOAT32)
TYPEMAP_ARRAYOUT1(float64_t,     NPY_FLOAT64)
TYPEMAP_ARRAYOUT1(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_ARRAYOUT1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARRAYOUT1

 /* Two dimensional input/output arrays */
%define TYPEMAP_ARRAYOUT2(type,typecode)
  %typemap(in) (type* ARRAYOUT_ARRAY2, int32_t DIM1, int32_t DIM2) (PyObject* temp=NULL) {
  temp = obj_to_array_no_conversion($input,typecode);
  if (!temp || !require_contiguous(temp)) SWIG_fail;
  $1 = (type*) PyArray_BYTES(temp);
  $2 = PyArray_DIM(temp,0);
  $3 = PyArray_DIM(temp,1);
}
%enddef

/* Define concrete examples of the TYPEMAP_ARRAYOUT2 macro */
TYPEMAP_ARRAYOUT2(bool,          NPY_BOOL)
TYPEMAP_ARRAYOUT2(char,          NPY_STRING)
TYPEMAP_ARRAYOUT2(uint8_t,       NPY_UINT8)
TYPEMAP_ARRAYOUT2(int16_t,       NPY_INT16)
TYPEMAP_ARRAYOUT2(uint16_t,      NPY_UINT16)
TYPEMAP_ARRAYOUT2(int32_t,       NPY_INT32)
TYPEMAP_ARRAYOUT2(uint32_t,      NPY_UINT32)
TYPEMAP_ARRAYOUT2(int64_t,       NPY_INT64)
TYPEMAP_ARRAYOUT2(uint64_t,      NPY_UINT64)
TYPEMAP_ARRAYOUT2(float32_t,     NPY_FLOAT32)
TYPEMAP_ARRAYOUT2(float64_t,     NPY_FLOAT64)
TYPEMAP_ARRAYOUT2(floatmax_t,    NPY_LONGDOUBLE)
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
 *     %apply (float64_t** ARGOUT_ARRAY1, {(float64_t** series, int32_t* len)}
 *     %apply (float64_t** ARGOUT_ARRAY2, {(float64_t** matrix, int32_t* d1, int32_t* d2)}
 *
 * with
 *
 *     void sum(float64_t* series, int32_t* len);
 *     void sum(float64_t** series, int32_t* len);
 *     void sum(float64_t** matrix, int32_t* d1, int32_t* d2);
 *
 * where sum mallocs the array and assigns dimensions and the pointer
 *
 */
%define TYPEMAP_ARGOUT1(type,typecode)
%typemap(in, numinputs=0) (type** ARGOUT1, int32_t* DIM1) {
    $1 = (type**) malloc(sizeof(type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (type** ARGOUT1, int32_t* DIM1) {
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

TYPEMAP_ARGOUT1(bool,          NPY_BOOL)
TYPEMAP_ARGOUT1(char,          NPY_STRING)
TYPEMAP_ARGOUT1(uint8_t,       NPY_UINT8)
TYPEMAP_ARGOUT1(int16_t,       NPY_INT16)
TYPEMAP_ARGOUT1(uint16_t,      NPY_UINT16)
TYPEMAP_ARGOUT1(int32_t,       NPY_INT32)
TYPEMAP_ARGOUT1(uint32_t,      NPY_UINT32)
TYPEMAP_ARGOUT1(int64_t,       NPY_INT64)
TYPEMAP_ARGOUT1(uint64_t,      NPY_UINT64)
TYPEMAP_ARGOUT1(float32_t,     NPY_FLOAT32)
TYPEMAP_ARGOUT1(float64_t,     NPY_FLOAT64)
TYPEMAP_ARGOUT1(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_ARGOUT1(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARGOUT1

%define TYPEMAP_ARGOUT2(type,typecode)
%typemap(in, numinputs=0) (type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
    $1 = (type**) malloc(sizeof(type*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
    $3 = (int32_t*) malloc(sizeof(int32_t));
}

%typemap(argout) (type** ARGOUT2, int32_t* DIM1, int32_t* DIM2) {
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

TYPEMAP_ARGOUT2(bool,          NPY_BOOL)
TYPEMAP_ARGOUT2(char,          NPY_STRING)
TYPEMAP_ARGOUT2(uint8_t,       NPY_UINT8)
TYPEMAP_ARGOUT2(int16_t,       NPY_INT16)
TYPEMAP_ARGOUT2(uint16_t,      NPY_UINT16)
TYPEMAP_ARGOUT2(int32_t,       NPY_INT32)
TYPEMAP_ARGOUT2(uint32_t,      NPY_UINT32)
TYPEMAP_ARGOUT2(int64_t,       NPY_INT64)
TYPEMAP_ARGOUT2(uint64_t,      NPY_UINT64)
TYPEMAP_ARGOUT2(float32_t,     NPY_FLOAT32)
TYPEMAP_ARGOUT2(float64_t,     NPY_FLOAT64)
TYPEMAP_ARGOUT2(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_ARGOUT2(PyObject,      NPY_OBJECT)

#undef TYPEMAP_ARGOUT2

/* Type mapping for grabbing a FILE * from Python */
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) ( FILE* ) {
    $1=0;
    if (PyFile_Check($input))
        $1=1;
}
%typemap(in) FILE* {
    if (!PyFile_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Need a file!");
        return NULL;
    }
    $1 = PyFile_AsFile($input);
}

/* input typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (shogun::SGString<type>* IN_STRINGS, int32_t NUM, int32_t MAXLEN)
{
    PyObject* list=(PyObject*) $input;

    $1=0;
    if (list && PyList_Check(list) && PyList_Size(list)>0)
    {
        $1=1;
        int32_t size=PyList_Size(list);
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (typecode == NPY_STRING)
            {
                if (!PyString_Check(o))
                {
                    $1=0;
                    break;
                }
            }
            else
            {
                if (!is_array(o) || array_dimensions(o)!=1 || array_type(o) != typecode)
                {
                    $1=0;
                    break;
                }
            }
        }
    }
}
%typemap(in) (shogun::SGString<type>* IN_STRINGS, int32_t NUM, int32_t MAXLEN) {
    PyObject* list=(PyObject*) $input;
    /* Check if is a list */
    if (!list || PyList_Check(list) || PyList_Size(list)==0)
    {
        int32_t size=PyList_Size(list);
        shogun::SGString<type>* strings=new shogun::SGString<type>[size];

        int32_t max_len=0;
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (typecode == NPY_STRING)
            {
                if (PyString_Check(o))
                {
                    int32_t len=PyString_Size(o);
                    max_len=shogun::CMath::max(len,max_len);
                    const char* str=PyString_AsString(o);

                    strings[i].length=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=new type[len];
                        memcpy(strings[i].string, str, len);
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be strings");

                    for (int32_t j=0; j<i; j++)
                        delete[] strings[i].string;
                    delete[] strings;
                    SWIG_fail;
                }
            }
            else
            {
                if (is_array(o) && array_dimensions(o)==1 && array_type(o) == typecode)
                {
                    int is_new_object=0;
                    PyObject* array = make_contiguous(o, &is_new_object, 1, typecode);
                    if (!array)
                        SWIG_fail;

                    type* str=(type*) PyArray_BYTES(array);
                    int32_t len = PyArray_DIM(array,0);
                    max_len=shogun::CMath::max(len,max_len);

                    strings[i].length=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=new type[len];
                        memcpy(strings[i].string, str, len*sizeof(type));
                    }

                    if (is_new_object)
                        Py_DECREF(array);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be of same array type");

                    for (int32_t j=0; j<i; j++)
                        delete[] strings[i].string;
                    delete[] strings;
                    SWIG_fail;
                }
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
%enddef

TYPEMAP_STRINGFEATURES_IN(bool,          NPY_BOOL)
TYPEMAP_STRINGFEATURES_IN(char,          NPY_STRING)
TYPEMAP_STRINGFEATURES_IN(uint8_t,       NPY_UINT8)
TYPEMAP_STRINGFEATURES_IN(int16_t,       NPY_INT16)
TYPEMAP_STRINGFEATURES_IN(uint16_t,      NPY_UINT16)
TYPEMAP_STRINGFEATURES_IN(int32_t,       NPY_INT32)
TYPEMAP_STRINGFEATURES_IN(uint32_t,      NPY_UINT32)
TYPEMAP_STRINGFEATURES_IN(int64_t,       NPY_INT64)
TYPEMAP_STRINGFEATURES_IN(uint64_t,      NPY_UINT64)
TYPEMAP_STRINGFEATURES_IN(float32_t,     NPY_FLOAT32)
TYPEMAP_STRINGFEATURES_IN(float64_t,     NPY_FLOAT64)
TYPEMAP_STRINGFEATURES_IN(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_IN(PyObject,      NPY_OBJECT)

#undef TYPEMAP_STRINGFEATURES_IN

/* output typemap for CStringFeatures */
%define TYPEMAP_STRINGFEATURES_ARGOUT(type,typecode)
%typemap(in, numinputs=0) (shogun::SGString<type>** ARGOUT_STRINGS, int32_t* NUM) {
    $1 = (shogun::SGString<type>**) malloc(sizeof(shogun::SGString<type>*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
}
%typemap(argout) (shogun::SGString<type>** ARGOUT_STRINGS, int32_t* NUM) {
    if (!$1 || !$2)
        SWIG_fail;

    shogun::SGString<type>* str=*$1;
    int32_t num=*$2;
    PyObject* list = PyList_New(num);

    if (list && str)
    {
        for (int32_t i=0; i<num; i++)
        {
            PyObject* s=NULL;

            if (typecode == NPY_STRING)
            {
                /* This path is only taking if str[i].string is a char*. However this cast is
                   required to build through for non char types. */
                s=PyString_FromStringAndSize((char*) str[i].string, str[i].length);
            }
            else
            {
                PyArray_Descr* descr=PyArray_DescrFromType(typecode);
                type* data = (type*) malloc(str[i].length*sizeof(type));
                if (descr && data)
                {
                    memcpy(data, str[i].string, str[i].length*sizeof(type));
                    npy_intp dims = str[i].length;

                    s = PyArray_NewFromDescr(&PyArray_Type,
                            descr, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
                    ((PyArrayObject*) s)->flags |= NPY_OWNDATA;
                }
                else
                    SWIG_fail;
            }

            PyList_SetItem(list, i, s);
            delete[] str[i].string;
        }
        delete[] str;
        $result = list;
    }
    else
        SWIG_fail;

    free($1); free($2);
}
%enddef

TYPEMAP_STRINGFEATURES_ARGOUT(bool,          NPY_BOOL)
TYPEMAP_STRINGFEATURES_ARGOUT(char,          NPY_STRING)
TYPEMAP_STRINGFEATURES_ARGOUT(uint8_t,       NPY_UINT8)
TYPEMAP_STRINGFEATURES_ARGOUT(int16_t,       NPY_INT16)
TYPEMAP_STRINGFEATURES_ARGOUT(uint16_t,      NPY_UINT16)
TYPEMAP_STRINGFEATURES_ARGOUT(int32_t,       NPY_INT32)
TYPEMAP_STRINGFEATURES_ARGOUT(uint32_t,      NPY_UINT32)
TYPEMAP_STRINGFEATURES_ARGOUT(int64_t,       NPY_INT64)
TYPEMAP_STRINGFEATURES_ARGOUT(uint64_t,      NPY_UINT64)
TYPEMAP_STRINGFEATURES_ARGOUT(float32_t,     NPY_FLOAT32)
TYPEMAP_STRINGFEATURES_ARGOUT(float64_t,     NPY_FLOAT64)
TYPEMAP_STRINGFEATURES_ARGOUT(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_STRINGFEATURES_ARGOUT(PyObject,      NPY_OBJECT)
#undef TYPEMAP_STRINGFEATURES_ARGOUT

/* input typemap for Sparse Features */
%define TYPEMAP_SPARSEFEATURES_IN(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER)
        (shogun::SGSparseVector<type>* IN_SPARSE, int32_t DIM1, int32_t DIM2)
{
    $1 = ( PyObject_HasAttrString($input, "indptr") &&
            PyObject_HasAttrString($input, "indices") &&
            PyObject_HasAttrString($input, "data") &&
            PyObject_HasAttrString($input, "shape")
         ) ? 1 : 0;
}

%typemap(in) (shogun::SGSparseVector<type>* IN_SPARSE, int32_t DIM1, int32_t DIM2)
{
    PyObject* o=(PyObject*) $input;

    /* a column compressed storage sparse matrix in python scipy
       looks like this

       A = csc_matrix( ... )
       A.indptr # pointer array
       A.indices # indices array
       A.data # nonzero values array
       A.shape # size of matrix

       >>> type(A.indptr)
       <type 'numpy.ndarray'> #int32
       >>> type(A.indices)
       <type 'numpy.ndarray'> #int32
       >>> type(A.data)
       <type 'numpy.ndarray'>
       >>> type(A.shape)
       <type 'tuple'>
     */

    if ( PyObject_HasAttrString(o, "indptr") &&
            PyObject_HasAttrString(o, "indices") &&
            PyObject_HasAttrString(o, "data") &&
            PyObject_HasAttrString(o, "shape"))
    {
        /* fetch sparse attributes */
        PyObject* indptr = PyObject_GetAttrString(o, "indptr");
        PyObject* indices = PyObject_GetAttrString(o, "indices");
        PyObject* data = PyObject_GetAttrString(o, "data");
        PyObject* shape = PyObject_GetAttrString(o, "shape");

        /* check that types are OK */
        if ((!is_array(indptr)) || (array_dimensions(indptr)!=1) ||
                (array_type(indptr)!=NPY_INT && array_type(indptr)!=NPY_LONG))
        {
            PyErr_SetString(PyExc_TypeError,"indptr array should be 1d int's");
            return NULL;
        }

        if (!is_array(indices) || array_dimensions(indices)!=1 ||
                (array_type(indices)!=NPY_INT && array_type(indices)!=NPY_LONG))
        {
            PyErr_SetString(PyExc_TypeError,"indices array should be 1d int's");
            return NULL;
        }

        if (!is_array(data) || array_dimensions(data)!=1 || array_type(data) != typecode)
        {
            PyErr_SetString(PyExc_TypeError,"data array should be 1d and match datatype");
            return NULL;
        }

        if (!PyTuple_Check(shape))
        {
            PyErr_SetString(PyExc_TypeError,"shape should be a tuple");
            return NULL;
        }

        /* get array dimensions */
        int32_t num_feat=PyInt_AsLong(PyTuple_GetItem(shape, 0));
        int32_t num_vec=PyInt_AsLong(PyTuple_GetItem(shape, 1));

        /* get indptr array */
        int is_new_object_indptr=0;
        PyObject* array_indptr = make_contiguous(indptr, &is_new_object_indptr, 1, NPY_INT32);
        if (!array_indptr) SWIG_fail;
        int32_t* bytes_indptr=(int32_t*) PyArray_BYTES(array_indptr);
        int32_t len_indptr = PyArray_DIM(array_indptr,0);

        /* get indices array */
        int is_new_object_indices=0;
        PyObject* array_indices = make_contiguous(indices, &is_new_object_indices, 1, NPY_INT32);
        if (!array_indices) SWIG_fail;
        int32_t* bytes_indices=(int32_t*) PyArray_BYTES(array_indices);
        int32_t len_indices = PyArray_DIM(array_indices,0);

        /* get data array */
        int is_new_object_data=0;
        PyObject* array_data = make_contiguous(data, &is_new_object_data, 1, typecode);
        if (!array_data) SWIG_fail;
        type* bytes_data=(type*) PyArray_BYTES(array_data);
        int32_t len_data = PyArray_DIM(array_data,0);

        if (len_indices!=len_data)
            SWIG_fail;

        shogun::SGSparseVector<type>* sfm = new shogun::SGSparseVector<type>[num_vec];

        for (int32_t i=0; i<num_vec; i++)
        {
            sfm[i].vec_index = i;
            sfm[i].num_feat_entries = 0;
            sfm[i].features = NULL;
        }

        for (int32_t i=1; i<len_indptr; i++)
        {
            int32_t num = bytes_indptr[i]-bytes_indptr[i-1];
            
            if (num>0)
            {
                shogun::SGSparseVectorEntry<type>* features=new shogun::SGSparseVectorEntry<type>[num];

                for (int32_t j=0; j<num; j++)
                {
                    features[j].feat_index=*bytes_indices;
                    features[j].entry=*bytes_data;

                    bytes_indices++;
                    bytes_data++;
                }
                sfm[i-1].num_feat_entries=num;
                sfm[i-1].features=features;
            }
        }

        if (is_new_object_indptr)
            Py_DECREF(array_indptr);
        if (is_new_object_indices)
            Py_DECREF(array_indices);
        if (is_new_object_data)
            Py_DECREF(array_data);

        Py_DECREF(indptr);
        Py_DECREF(indices);
        Py_DECREF(data);
        Py_DECREF(shape);

        $1=sfm;
        $2=num_feat;
        $3=num_vec;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a column compressed sparse matrix");
        return NULL;
    }
}
%enddef

TYPEMAP_SPARSEFEATURES_IN(bool,          NPY_BOOL)
TYPEMAP_SPARSEFEATURES_IN(char,          NPY_STRING)
TYPEMAP_SPARSEFEATURES_IN(uint8_t,       NPY_UINT8)
TYPEMAP_SPARSEFEATURES_IN(int16_t,       NPY_INT16)
TYPEMAP_SPARSEFEATURES_IN(uint16_t,      NPY_UINT16)
TYPEMAP_SPARSEFEATURES_IN(int32_t,       NPY_INT32)
TYPEMAP_SPARSEFEATURES_IN(uint32_t,      NPY_UINT32)
TYPEMAP_SPARSEFEATURES_IN(int64_t,       NPY_INT64)
TYPEMAP_SPARSEFEATURES_IN(uint64_t,      NPY_UINT64)
TYPEMAP_SPARSEFEATURES_IN(float32_t,     NPY_FLOAT32)
TYPEMAP_SPARSEFEATURES_IN(float64_t,     NPY_FLOAT64)
TYPEMAP_SPARSEFEATURES_IN(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_IN(PyObject,      NPY_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_IN

/* output typemap for sparse features returns (data, row, ptr) */
%define TYPEMAP_SPARSEFEATURES_ARGOUT(type,typecode)
%typemap(in, numinputs=0) (shogun::SGSparseVector<type>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {
    $1 = (shogun::SGSparseVector<type>**) malloc(sizeof(shogun::SGSparseVector<type>*));
    $2 = (int32_t*) malloc(sizeof(int32_t));
    $3 = (int32_t*) malloc(sizeof(int32_t));
    $4 = (int64_t*) malloc(sizeof(int64_t));
}
%typemap(argout) (shogun::SGSparseVector<type>** ARGOUT_SPARSE, int32_t* DIM1, int32_t* DIM2, int64_t* NNZ) {
    if (!$1 || !$2 || !$3 || !$4)
        SWIG_fail;

    shogun::SGSparseVector<type>* sfm=*$1;
    int32_t num_feat=*$2;
    int32_t num_vec=*$3;
    int64_t nnz=*$4;

    PyObject* tuple = PyTuple_New(3);

    if (tuple && sfm)
    {
        PyObject* data_py=NULL;
        PyObject* indices_py=NULL;
        PyObject* indptr_py=NULL;

        PyArray_Descr* descr=PyArray_DescrFromType(NPY_INT32);
        PyArray_Descr* descr_data=PyArray_DescrFromType(typecode);

        int32_t* indptr = (int32_t*) malloc((num_vec+1)*sizeof(int32_t));
        int32_t* indices = (int32_t*) malloc(nnz*sizeof(int32_t));
        type* data = (type*) malloc(nnz*sizeof(type));

        if (descr && descr_data && indptr && indices && data)
        {
            indptr[0]=0;

            int32_t* i_ptr=indices;
            type* d_ptr=data;

            for (int32_t i=0; i<num_vec; i++)
            {
                indptr[i+1]=indptr[i];
                if (sfm[i].vec_index==i)
                {
                    indptr[i+1]+=sfm[i].num_feat_entries;

                    for (int32_t j=0; j<sfm[i].num_feat_entries; j++)
                    {
                        *i_ptr=sfm[i].features[j].feat_index;
                        *d_ptr=sfm[i].features[j].entry;

                        i_ptr++;
                        d_ptr++;
                    }
                }
            }

            npy_intp indptr_dims = num_vec+1;
            indptr_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &indptr_dims, NULL, (void*) indptr, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) indptr_py)->flags |= NPY_OWNDATA;

            npy_intp dims = nnz;
            indices_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &dims, NULL, (void*) indices, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) indices_py)->flags |= NPY_OWNDATA;

            data_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr_data, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
            ((PyArrayObject*) data_py)->flags |= NPY_OWNDATA;

            PyTuple_SetItem(tuple, 0, data_py);
            PyTuple_SetItem(tuple, 1, indices_py);
            PyTuple_SetItem(tuple, 2, indptr_py);
            $result = tuple;
        }
        else
            SWIG_fail;
    }
    else
        SWIG_fail;

    free($1); free($2); free($3); free($4);
}
%enddef

TYPEMAP_SPARSEFEATURES_ARGOUT(bool,          NPY_BOOL)
TYPEMAP_SPARSEFEATURES_ARGOUT(char,          NPY_STRING)
TYPEMAP_SPARSEFEATURES_ARGOUT(uint8_t,       NPY_UINT8)
TYPEMAP_SPARSEFEATURES_ARGOUT(int16_t,       NPY_INT16)
TYPEMAP_SPARSEFEATURES_ARGOUT(uint16_t,      NPY_UINT16)
TYPEMAP_SPARSEFEATURES_ARGOUT(int32_t,       NPY_INT32)
TYPEMAP_SPARSEFEATURES_ARGOUT(uint32_t,      NPY_UINT32)
TYPEMAP_SPARSEFEATURES_ARGOUT(int64_t,       NPY_INT64)
TYPEMAP_SPARSEFEATURES_ARGOUT(uint64_t,      NPY_UINT64)
TYPEMAP_SPARSEFEATURES_ARGOUT(float32_t,     NPY_FLOAT32)
TYPEMAP_SPARSEFEATURES_ARGOUT(float64_t,     NPY_FLOAT64)
TYPEMAP_SPARSEFEATURES_ARGOUT(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_ARGOUT(PyObject,      NPY_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_ARGOUT

#endif
