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
#include <shogun/lib/DataType.h>

#undef _POSIX_C_SOURCE
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
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

void* get_copy(void* src, size_t len)
{
    void* copy=SG_MALLOC(uint8_t, len);
    memcpy(copy, src, len);
    return copy;
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
    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);
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

/* One dimensional output arrays */
%define TYPEMAP_OUT_SGVECTOR(type,typecode)
%typemap(out) shogun::SGVector<type>
{
    npy_intp dims= (npy_intp) $1.vlen;
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr)
    {
        void* copy=get_copy($1.vector, sizeof(type)*size_t($1.vlen));
        $result = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) $result)->flags |= NPY_OWNDATA;
    }

    $1.free_vector();

    if (!descr)
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

    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);
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

/* Two dimensional output arrays */
%define TYPEMAP_OUT_SGMATRIX(type,typecode)
%typemap(out) shogun::SGMatrix<type>
{
    npy_intp dims[2]= {(npy_intp) $1.num_rows, (npy_intp) $1.num_cols };
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr)
    {
        void* copy=get_copy($1.matrix, sizeof(type)*size_t($1.num_rows)*size_t($1.num_cols));
        $result = PyArray_NewFromDescr(&PyArray_Type,
            descr, 2, dims, NULL, (void*) copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) $result)->flags |= NPY_OWNDATA;
    }

    $1.free_matrix();

    if (!descr)
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
    int is_new_object;
    PyObject* array = make_contiguous($input, &is_new_object, -1,typecode, true);
    if (!array)
        SWIG_fail;

    int32_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      SWIG_fail;

    int32_t* temp_dims = SG_MALLOC(int32_t, ndim);

    npy_intp* py_dims = PyArray_DIMS(array);

    for (int32_t i=0; i<ndim; i++)
      temp_dims[i] = py_dims[i];
    
    $1 = SGNDArray<type>((type*) PyArray_BYTES(array), temp_dims, ndim);

    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);
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
        shogun::SGString<type>* strings=SG_MALLOC(shogun::SGString<type>, size);

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

                    strings[i].slen=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=SG_MALLOC(type, len);
                        memcpy(strings[i].string, str, len);
                    }
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be strings");

                    for (int32_t j=0; j<i; j++)
                        SG_FREE(strings[i].string);
                    SG_FREE(strings);
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

                    strings[i].slen=len;
                    strings[i].string=NULL;

                    if (len>0)
                    {
                        strings[i].string=SG_MALLOC(type, len);
                        memcpy(strings[i].string, str, len*sizeof(type));
                    }

                    if (is_new_object)
                        Py_DECREF(array);
                }
                else
                {
                    PyErr_SetString(PyExc_TypeError, "all elements in list must be of same array type");

                    for (int32_t j=0; j<i; j++)
                        SG_FREE(strings[i].string);
                    SG_FREE(strings);
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
%typemap(out) shogun::SGStringList<type>
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
                s=PyString_FromStringAndSize((char*) str[i].string, str[i].slen);
            }
            else
            {
                PyArray_Descr* descr=PyArray_DescrFromType(typecode);
                type* data = (type*) malloc(str[i].slen*sizeof(type));
                if (descr && data)
                {
                    memcpy(data, str[i].string, str[i].slen*sizeof(type));
                    npy_intp dims = str[i].slen;

                    s = PyArray_NewFromDescr(&PyArray_Type,
                            descr, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
                    ((PyArrayObject*) s)->flags |= NPY_OWNDATA;
                }
                else
                    SWIG_fail;
            }

            PyList_SetItem(list, i, s);
        }
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

        shogun::SGSparseVector<type>* sfm = SG_MALLOC(shogun::SGSparseVector<type>, num_vec);

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
                shogun::SGSparseVectorEntry<type>* features=SG_MALLOC(shogun::SGSparseVectorEntry<type>, num);

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
%typemap(out) shogun::SGSparseMatrix<type>
    
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
#endif /* HAVE_PYTHON */
