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
 * Copyright (C) 2012 Evgeniy Andreev (gsomix)
 */

%include "DenseFeatures_protocols.i"
%include "CustomKernel_protocols.i"
%include "DenseLabels_protocols.i"
%include "SGVector_protocols.i"

#ifdef HAVE_PYTHON
%{
#include <stdio.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/memory.h>

#undef _POSIX_C_SOURCE
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

/* Functions to extract array attributes.
 */
static bool is_array(PyObject* a) { return (a) && PyArray_Check(a); }
static int array_type(PyObject* a) { return (int) PyArray_TYPE(a); }
static int array_dimensions(PyObject* a)  { return ((PyArrayObject *)a)->nd; }
static int array_size(PyObject* a, int i) { return ((PyArrayObject *)a)->dimensions[i]; }
static bool array_is_contiguous(PyObject* a) { return PyArray_ISCONTIGUOUS(a); }

/* Given a PyObject, return a string describing its type.
 */
static const char* typecode_string(PyObject* py_obj) {
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable";

#if PY_VERSION_HEX >= 0x03000000
  if (PyUnicode_Check( py_obj)) return "unicode";
#else
  if (PyString_Check(  py_obj)) return "string";
#endif

#if PY_VERSION_HEX >= 0x03000000
  if (PyLong_Check(    py_obj)) return "int";
#else
  if (PyInt_Check(     py_obj)) return "int";
#endif
  if (PyFloat_Check(   py_obj)) return "float";
  if (PyDict_Check(    py_obj)) return "dict";
  if (PyList_Check(    py_obj)) return "list";
  if (PyTuple_Check(   py_obj)) return "tuple";
  if (PyModule_Check(  py_obj)) return "module";

#if PY_VERSION_HEX < 0x03000000
  if (PyFile_Check(    py_obj)) return "file";
  if (PyInstance_Check(py_obj)) return "instance";
#endif

  return "unknown type";
}

static const char* typecode_string(int typecode) {
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

/* Given a PyArrayObject, check to see if it is contiguous.  If so,
 * return the input pointer and flag it as not a new object.  If it is
 * not contiguous, create a new PyArrayObject using the original data,
 * flag it as a new object and return the pointer.
 *
 * If array is NULL or dimensionality or typecode does not match
 * return NULL
 */
static PyObject* make_contiguous(PyObject* ary, int* is_new_object,
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

    if (!::is_array(array))
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

static int is_pyvector(PyObject* obj, int typecode)
{
    return  ((obj && !PyList_Check(obj)) &&
            (
             ::is_array(obj) &&
             array_dimensions(obj)==1 &&
             (array_type(obj) == typecode || PyArray_EquivTypenums(array_type(obj), typecode))
            )) ? 1 : 0;
}

static int is_pymatrix(PyObject* obj, int typecode)
{
    return ((obj && !PyList_Check(obj)) &&
                (
				 ::is_array(obj) &&
                 array_dimensions(obj)==2 &&
                 (array_type(obj) == typecode || PyArray_EquivTypenums(array_type(obj), typecode))
                )) ? 1 : 0;
}

static int is_pyarray(PyObject* obj, int typecode)
{
    return ((obj && !PyList_Check(obj)) &&
                (
				 ::is_array(obj) &&
                 (array_type(obj) == typecode || PyArray_EquivTypenums(array_type(obj), typecode))
                )) ? 1 : 0;
}

static int is_pysparse_matrix(PyObject* obj, int typecode)
{
    return ( obj && PyObject_HasAttrString(obj, "indptr") &&
            PyObject_HasAttrString(obj, "indices") &&
            PyObject_HasAttrString(obj, "data") &&
            PyObject_HasAttrString(obj, "shape")
         ) ? 1 : 0;
}

static int is_pystring_list(PyObject* obj, int typecode)
{
    PyObject* list=(PyObject*) obj;

    int result=0;
    if (list && PyList_Check(list) && PyList_Size(list)>0)
    {
        result=1;
        int32_t size=PyList_Size(list);
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);

            if (typecode == NPY_STRING || typecode == NPY_UNICODE)
            {
#if PY_VERSION_HEX >= 0x03000000
                if (!PyUnicode_Check(o))
#else
				if (!PyString_Check(o))
#endif
                {
                    result=0;
                    break;
                }
            }
            else
            {
                if (!::is_array(o) || array_dimensions(o)!=1 || array_type(o) != typecode)
                {
                    result=0;
                    break;
                }
            }
        }
    }

    return result;
}


template <class type>
static bool vector_from_numpy(SGVector<type>& sg_vec, PyObject* obj, int typecode)
{
    if (!is_pyvector(obj, typecode))
    {
        PyErr_SetString(PyExc_TypeError,"not a numpy vector of appropriate type");
        return false;
    }

    int is_new_object;
    PyObject* array = make_contiguous(obj, &is_new_object, 1,typecode, true);
    if (!array)
        return false;

    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    type* vec = (type*) PyArray_BYTES(array);
    int32_t vlen = PyArray_DIM(array,0);
    Py_DECREF(array);

    sg_vec=shogun::SGVector<type>(vec, vlen);

    return true;
}

template <class type>
static bool vector_to_numpy(PyObject* &obj, SGVector<type> sg_vec, int typecode)
{
    npy_intp dims= (npy_intp) sg_vec.vlen;
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr)
    {
        void* copy=get_copy(sg_vec.vector, sizeof(type)*size_t(sg_vec.vlen));
        obj = PyArray_NewFromDescr(&PyArray_Type,
                descr, 1, &dims, NULL, copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) obj)->flags |= NPY_OWNDATA;
    }

    return descr!=NULL;
}

template <class type>
static bool matrix_from_numpy(SGMatrix<type>& sg_matrix, PyObject* obj, int typecode)
{
    if (!is_pymatrix(obj, typecode))
    {
        PyErr_SetString(PyExc_TypeError,"not a numpy matrix of appropriate type");
        return false;
    }

    int is_new_object;
    PyObject* array = make_contiguous(obj, &is_new_object, 2,typecode, true);
    if (!array)
        return false;

    sg_matrix = shogun::SGMatrix<type>((type*) PyArray_BYTES(array),
            PyArray_DIM(array,0), PyArray_DIM(array,1), true);

    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);

    return true;
}

template <class type>
static bool matrix_to_numpy(PyObject* &obj, SGMatrix<type> sg_matrix, int typecode)
{
    npy_intp dims[2]= {(npy_intp) sg_matrix.num_rows, (npy_intp) sg_matrix.num_cols };
    PyArray_Descr* descr=PyArray_DescrFromType(typecode);

    if (descr)
    {
        void* copy=get_copy(sg_matrix.matrix, sizeof(type)*size_t(sg_matrix.num_rows)*size_t(sg_matrix.num_cols));
        obj = PyArray_NewFromDescr(&PyArray_Type,
            descr, 2, dims, NULL, (void*) copy, NPY_FARRAY | NPY_WRITEABLE, NULL);
        ((PyArrayObject*) obj)->flags |= NPY_OWNDATA;
    }

    return descr!=NULL;
}

template <class type>
static bool array_from_numpy(SGNDArray<type>& sg_array, PyObject* obj, int typecode)
{
    if (!is_pyarray(obj, typecode))
    {
        PyErr_SetString(PyExc_TypeError,"not a nd-array");
        return false;
    }

    int is_new_object;
    PyObject* array = make_contiguous(obj, &is_new_object, -1,typecode, true);
    if (!array)
        return false;

    int32_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      return false;

    int32_t* temp_dims = SG_MALLOC(int32_t, ndim);

    npy_intp* py_dims = PyArray_DIMS(array);

    for (int32_t i=0; i<ndim; i++)
      temp_dims[i] = py_dims[i];

    sg_array = SGNDArray<type>((type*) PyArray_BYTES(array), temp_dims, ndim);

    ((PyArrayObject*) array)->flags &= (-1 ^ NPY_OWNDATA);
    Py_DECREF(array);

    return true;
}

template <class type>
static bool string_from_strpy(SGStringList<type>& sg_strings, PyObject* obj, int typecode)
{
    PyObject* list=(PyObject*) obj;

    /* Check if is a list */
    if (!list || PyList_Check(list) || PyList_Size(list)==0)
    {
        int32_t size=PyList_Size(list);
        shogun::SGString<type>* strings=SG_MALLOC(shogun::SGString<type>, size);

        int32_t max_len=0;
        for (int32_t i=0; i<size; i++)
        {
            PyObject *o = PyList_GetItem(list,i);
            if (typecode == NPY_STRING || typecode == NPY_UNICODE)
            {
#if PY_VERSION_HEX >= 0x03000000
                if (PyUnicode_Check(o))
#else
				if (PyString_Check(o))
#endif
                {

#if PY_VERSION_HEX >= 0x03000000
					int32_t len = PyUnicode_GetSize((PyObject*) o);
    				const char* str = PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<PyObject*>(o)));
#else
                    int32_t len = PyString_Size(o);
                    const char* str = PyString_AsString(o);
#endif
					max_len=shogun::CMath::max(len,max_len);

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
                    return false;
                }
            }
            else
            {
                if (::is_array(o) && array_dimensions(o)==1 && array_type(o) == typecode)
                {
                    int is_new_object=0;
                    PyObject* array = make_contiguous(o, &is_new_object, 1, typecode);
                    if (!array)
                        return false;

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
                    return false;
                }
            }
        }

        SGStringList<type> sl;
        sl.strings=strings;
        sl.num_strings=size;
        sl.max_string_length=max_len;
        sg_strings=sl;

        return true;
    }
    else
    {
        PyErr_SetString(PyExc_TypeError,"not a/empty list");
        return false;
    }
}

template <class type>
static bool string_to_strpy(PyObject* &obj, SGStringList<type> sg_strings, int typecode)
{
    shogun::SGString<type>* str=sg_strings.strings;
    int32_t num=sg_strings.num_strings;
    PyObject* list = PyList_New(num);

    if (list && str)
    {
        for (int32_t i=0; i<num; i++)
        {
            PyObject* s=NULL;

            if (typecode == NPY_STRING || typecode == NPY_UNICODE)
            {
                /* This path is only taking if str[i].string is a char*. However this cast is
                   required to build through for non char types. */
#if PY_VERSION_HEX >= 0x03000000
				s=PyUnicode_FromStringAndSize((char*) str[i].string, str[i].slen);
#else
				s=PyString_FromStringAndSize((char*) str[i].string, str[i].slen);
#endif
            }
            else
            {
                PyArray_Descr* descr=PyArray_DescrFromType(typecode);
                type* data = SG_MALLOC(type, str[i].slen);
                if (descr && data)
                {
                    memcpy(data, str[i].string, str[i].slen*sizeof(type));
                    npy_intp dims = str[i].slen;

                    s = PyArray_NewFromDescr(&PyArray_Type,
                            descr, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
                    ((PyArrayObject*) s)->flags |= NPY_OWNDATA;
                }
                else
                    return false;
            }

            PyList_SetItem(list, i, s);
        }
        obj = list;
        return true;
    }
    else
        return false;
}

template <class type>
static bool spmatrix_from_numpy(SGSparseMatrix<type>& sg_matrix, PyObject* obj, int typecode)
{
    PyObject* o=(PyObject*) obj;

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
    if (!is_pysparse_matrix(o, typecode))
    {
        PyErr_SetString(PyExc_TypeError,"not a column compressed sparse matrix");
        return false;
    }

    /* fetch sparse attributes */
    PyObject* indptr = PyObject_GetAttrString(o, "indptr");
    PyObject* indices = PyObject_GetAttrString(o, "indices");
    PyObject* data = PyObject_GetAttrString(o, "data");
    PyObject* shape = PyObject_GetAttrString(o, "shape");

    /* check that types are OK */
    if ((!::is_array(indptr)) || (array_dimensions(indptr)!=1) ||
            (array_type(indptr)!=NPY_INT && array_type(indptr)!=NPY_LONG))
    {
        PyErr_SetString(PyExc_TypeError,"indptr array should be 1d int's");
        return false;
    }

    if (!::is_array(indices) || array_dimensions(indices)!=1 ||
            (array_type(indices)!=NPY_INT && array_type(indices)!=NPY_LONG))
    {
        PyErr_SetString(PyExc_TypeError,"indices array should be 1d int's");
        return false;
    }

    if (!::is_array(data) || array_dimensions(data)!=1 || array_type(data) != typecode)
    {
        PyErr_SetString(PyExc_TypeError,"data array should be 1d and match datatype");
        return false;
    }

    if (!PyTuple_Check(shape))
    {
        PyErr_SetString(PyExc_TypeError,"shape should be a tuple");
        return false;
    }

    /* get array dimensions */
#if PY_VERSION_HEX >= 0x03000000
    int32_t num_feat=PyLong_AsLong(PyTuple_GetItem(shape, 0));
    int32_t num_vec=PyLong_AsLong(PyTuple_GetItem(shape, 1));
#else
    int32_t num_feat=PyInt_AsLong(PyTuple_GetItem(shape, 0));
    int32_t num_vec=PyInt_AsLong(PyTuple_GetItem(shape, 1));
#endif

    /* get indptr array */
    int is_new_object_indptr=0;
    PyObject* array_indptr = make_contiguous(indptr, &is_new_object_indptr, 1, NPY_INT32);
    if (!array_indptr) return false;
    int32_t* bytes_indptr=(int32_t*) PyArray_BYTES(array_indptr);
    int32_t len_indptr = PyArray_DIM(array_indptr,0);

    /* get indices array */
    int is_new_object_indices=0;
    PyObject* array_indices = make_contiguous(indices, &is_new_object_indices, 1, NPY_INT32);
    if (!array_indices) return false;
    int32_t* bytes_indices=(int32_t*) PyArray_BYTES(array_indices);
    int32_t len_indices = PyArray_DIM(array_indices,0);

    /* get data array */
    int is_new_object_data=0;
    PyObject* array_data = make_contiguous(data, &is_new_object_data, 1, typecode);
    if (!array_data) return false;
    type* bytes_data=(type*) PyArray_BYTES(array_data);
    int32_t len_data = PyArray_DIM(array_data,0);

    if (len_indices!=len_data)
        return false;

    shogun::SGSparseVector<type>* sfm = SG_MALLOC(shogun::SGSparseVector<type>, num_vec);

    for (int32_t i=0; i<num_vec; i++)
        new (&sfm[i]) SGSparseVector<type>();

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
    sg_matrix=sm;

    return true;
}

template <class type>
static bool spmatrix_to_numpy(PyObject* &obj, SGSparseMatrix<type> sg_matrix, int typecode)
{
    shogun::SGSparseVector<type>* sfm=sg_matrix.sparse_matrix;
    int32_t num_feat=sg_matrix.num_features;
    int32_t num_vec=sg_matrix.num_vectors;

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

        int32_t* indptr = SG_MALLOC(int32_t, num_vec+1);
        int32_t* indices = SG_MALLOC(int32_t, nnz);
        type* data = SG_MALLOC(type, nnz);

        if (descr && descr_data && indptr && indices && data)
        {
            indptr[0]=0;

            int32_t* i_ptr=indices;
            type* d_ptr=data;

            for (int32_t i=0; i<num_vec; i++)
            {
                indptr[i+1]=indptr[i];
                indptr[i+1]+=sfm[i].num_feat_entries;

                for (int32_t j=0; j<sfm[i].num_feat_entries; j++)
                {
                    *i_ptr=sfm[i].features[j].feat_index;
                    *d_ptr=sfm[i].features[j].entry;

                    i_ptr++;
                    d_ptr++;
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
            obj = tuple;
            return true;
        }
        else
            return false;
    }
    else
        return false;
}

template <class type>
static bool spvector_to_numpy(PyObject* &obj, SGSparseVector<type> sg_vector, int typecode)
{
    PyObject* tuple = PyTuple_New(2);
    npy_intp dims = sg_vector.num_feat_entries;

    if (!tuple)
        return false;

    PyObject* data_py=NULL;
    PyObject* indices_py=NULL;

    PyArray_Descr* descr=PyArray_DescrFromType(NPY_INT32);
    PyArray_Descr* descr_data=PyArray_DescrFromType(typecode);

    int32_t* indices = SG_MALLOC(int32_t, dims);
    type* data = SG_MALLOC(type, dims);

    if (!(descr && descr_data && indices && data))
        return false;

    int32_t* i_ptr=indices;
    type* d_ptr=data;

    for (int32_t j=0; j<sg_vector.num_feat_entries; j++)
    {
        *i_ptr=sg_vector.features[j].feat_index;
        *d_ptr=sg_vector.features[j].entry;

        i_ptr++;
        d_ptr++;
    }

    indices_py = PyArray_NewFromDescr(&PyArray_Type,
            descr, 1, &dims, NULL, (void*) indices, NPY_FARRAY | NPY_WRITEABLE, NULL);
    ((PyArrayObject*) indices_py)->flags |= NPY_OWNDATA;

    data_py = PyArray_NewFromDescr(&PyArray_Type,
            descr_data, 1, &dims, NULL, (void*) data, NPY_FARRAY | NPY_WRITEABLE, NULL);
    ((PyArrayObject*) data_py)->flags |= NPY_OWNDATA;

    PyTuple_SetItem(tuple, 0, data_py);
    PyTuple_SetItem(tuple, 1, indices_py);
    obj = tuple;
    return true;
}

%}

/* CFeatures to ... */
%define FEATURES_BY_TYPECODE(obj, f, type, typecode)
	switch (typecode) {
	case F_BOOL:
		obj=SWIG_NewPointerObj(f, $descriptor(type<bool> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_CHAR:
		obj=SWIG_NewPointerObj(f, $descriptor(type<char> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_BYTE:
		obj=SWIG_NewPointerObj(f, $descriptor(type<uint8_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_SHORT:
		obj=SWIG_NewPointerObj(f, $descriptor(type<int16_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_WORD:
		obj=SWIG_NewPointerObj(f, $descriptor(type<uint16_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_INT:
		obj=SWIG_NewPointerObj(f, $descriptor(type<int32_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_UINT:
		obj=SWIG_NewPointerObj(f, $descriptor(type<uint32_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_LONG:
		obj=SWIG_NewPointerObj(f, $descriptor(type<int64_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_ULONG:
		obj=SWIG_NewPointerObj(f, $descriptor(type<uint64_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_SHORTREAL:
		obj=SWIG_NewPointerObj(f, $descriptor(type<float32_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_DREAL:
		obj=SWIG_NewPointerObj(f, $descriptor(type<float64_t> *), SWIG_POINTER_EXCEPTION);
		break;
	case F_LONGREAL:
		obj=SWIG_NewPointerObj(f, $descriptor(type<floatmax_t> *), SWIG_POINTER_EXCEPTION);
		break;
	default:
		obj=SWIG_NewPointerObj(f, $descriptor(shogun::CFeatures*), SWIG_POINTER_EXCEPTION);
		break;
	}
%enddef

%typemap(out) shogun::CFeatures*
{
	int feats_class=$1->get_feature_class();
	int feats_type=$1->get_feature_type();

	switch (feats_class){
	case C_DENSE:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CDenseFeatures, feats_type)
		break;
	}

	case C_SPARSE:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CSparseFeatures, feats_type)
		break;
	}

	case C_STRING:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CStringFeatures, feats_type)
		break;
	}

	case C_COMBINED:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CCombinedFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_COMBINED_DOT:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CCombinedDotFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_WD:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CWDFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_SPEC:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CExplicitSpecFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_WEIGHTEDSPEC:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CImplicitWeightedSpecFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_POLY:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CPolyFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_STREAMING_DENSE:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingDenseFeatures, feats_type)
		break;
	}

	case C_STREAMING_SPARSE:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingSparseFeatures, feats_type)
		break;
	}

	case C_STREAMING_STRING:
	{
		FEATURES_BY_TYPECODE($result, $1, shogun::CStreamingStringFeatures, feats_type)
		break;
	}
	case C_STREAMING_VW:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CStreamingVwFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_BINNED_DOT:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CBinnedDotFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	case C_DIRECTOR_DOT:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CDirectorDotFeatures*), SWIG_POINTER_EXCEPTION);
		break;

	default:
		$result=SWIG_NewPointerObj($1, $descriptor(shogun::CFeatures*), SWIG_POINTER_EXCEPTION);
		break;
	}
}

#ifdef PYTHON3
%typemap(typecheck, precedence=SWIG_TYPECHECK_STRING) const char *
{
	$1 = (PyUnicode_Check($input) || PyString_Check($input)) ? 1 : 0;
}
%typemap(in) const char *
{
	if (PyString_Check($input))
	{
		$1 = PyString_AsString($input);
	}
	else if (PyUnicode_Check($input))
	{
		$1 = PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<PyObject*>($input)));
    	}
	else
	{
		PyErr_SetString(PyExc_TypeError, "Expected a string");
	}
}
%typemap(freearg) const char *
{
	// nothing to do there
}
#endif

/* One dimensional input arrays */
%define TYPEMAP_IN_SGVECTOR(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<type>
{
    $1 = is_pyvector($input, typecode);
}

%typemap(in) shogun::SGVector<type>
{
    if (!vector_from_numpy<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGVECTOR macros */
TYPEMAP_IN_SGVECTOR(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_IN_SGVECTOR(char,          NPY_UNICODE)
#else
TYPEMAP_IN_SGVECTOR(char,          NPY_STRING)
#endif
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
    if (!vector_to_numpy($result, $1, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGVECTOR macros */
TYPEMAP_OUT_SGVECTOR(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_OUT_SGVECTOR(char,          NPY_UNICODE)
#else
TYPEMAP_OUT_SGVECTOR(char,          NPY_STRING)
#endif
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
    $1 = is_pymatrix($input, typecode);
}

%typemap(in) shogun::SGMatrix<type>
{
    if (!matrix_from_numpy<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_IN_SGMATRIX macros */
TYPEMAP_IN_SGMATRIX(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_IN_SGMATRIX(char,          NPY_UNICODE)
#else
TYPEMAP_IN_SGMATRIX(char,          NPY_STRING)
#endif
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
    if (!matrix_to_numpy($result, $1, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUT_SGMATRIX macros */
TYPEMAP_OUT_SGMATRIX(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_OUT_SGMATRIX(char,          NPY_UNICODE)
#else
TYPEMAP_OUT_SGMATRIX(char,          NPY_STRING)
#endif
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
    $1 = is_pyarray($input, typecode);
}

%typemap(in) shogun::SGNDArray<type>
{
    if (! array_from_numpy<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_INND macros */
TYPEMAP_INND(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_INND(char,          NPY_UNICODE)
#else
TYPEMAP_INND(char,          NPY_STRING)
#endif
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
    $1 = is_pystring_list($input, typecode);
}
%typemap(in) shogun::SGStringList<type>
{
    if (! string_from_strpy<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

TYPEMAP_STRINGFEATURES_IN(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_STRINGFEATURES_IN(char,          NPY_UNICODE)
#else
TYPEMAP_STRINGFEATURES_IN(char,          NPY_STRING)
#endif
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
    if (!string_to_strpy($result, $1, typecode))
        SWIG_fail;
}
%enddef

TYPEMAP_STRINGFEATURES_OUT(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_STRINGFEATURES_OUT(char,          NPY_UNICODE)
#else
TYPEMAP_STRINGFEATURES_OUT(char,          NPY_STRING)
#endif
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
    $1 = is_pysparse_matrix($input, typecode);
}

%typemap(in) shogun::SGSparseMatrix<type>
{
    if (! spmatrix_from_numpy<type>($1, $input, typecode))
        SWIG_fail;
}
%enddef

TYPEMAP_SPARSEFEATURES_IN(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_SPARSEFEATURES_IN(char,          NPY_UNICODE)
#else
TYPEMAP_SPARSEFEATURES_IN(char,          NPY_STRING)
#endif
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
    if (!spvector_to_numpy($result, $1, typecode))
        SWIG_fail;
}

%typemap(out) shogun::SGSparseMatrix<type>
{
    if (!spmatrix_to_numpy($result, $1, typecode))
        SWIG_fail;
}
%enddef

TYPEMAP_SPARSEFEATURES_OUT(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_SPARSEFEATURES_OUT(char,          NPY_UNICODE)
#else
TYPEMAP_SPARSEFEATURES_OUT(char,          NPY_STRING)
#endif
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
