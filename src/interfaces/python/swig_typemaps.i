/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Viktor Gal, Sergey Lisitsyn
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
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
extern "C" {
#include <Python.h>
#include <numpy/arrayobject.h>
}

/* Functions to extract array attributes.
 */
static bool is_array(PyObject* a) { return (a) && PyArray_Check(a); }
static int array_type(const PyObject* a) { return (int) PyArray_TYPE((const PyArrayObject*)a); }
static int array_dimensions(const PyObject* a)  { return PyArray_NDIM((const PyArrayObject *)a); }

/* Given a PyObject, return a string describing its type.
 */
static const char* typecode_string(PyObject* py_obj) {
  if (py_obj == NULL          ) return "C NULL value";
  if (PyCallable_Check(py_obj)) return "callable";
  if (PyUnicode_Check( py_obj)) return "unicode";
  if (PyLong_Check(    py_obj)) return "int";
  if (PyFloat_Check(   py_obj)) return "float";
  if (PyDict_Check(    py_obj)) return "dict";
  if (PyList_Check(    py_obj)) return "list";
  if (PyTuple_Check(   py_obj)) return "tuple";
  if (PyModule_Check(  py_obj)) return "module";

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
static PyArrayObject* make_contiguous(
    PyObject* ary, int* is_new_object, int dims, int typecode, bool force_copy=false)
{
    PyObject* array;
    if (PyArray_ISFARRAY((PyArrayObject*)ary) && !force_copy)
    {
        array = ary;
        *is_new_object = 0;
    }
    else
    {
        array=PyArray_FromAny((PyObject*)ary, NULL,0,0,  NPY_ARRAY_FARRAY|NPY_ARRAY_ENSURECOPY, NULL);
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

    return (PyArrayObject*)array;
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

template <class type>
static bool vector_from_numpy(SGVector<type>& sg_vec, PyObject* obj, int typecode)
{
    int is_new_object;
    PyArrayObject* array = make_contiguous(obj, &is_new_object, 1,typecode, true);
    if (!array)
        return false;

    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA);
    type* vec = (type*) PyArray_DATA(array);
    index_t vlen = PyArray_DIM(array,0);
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
                descr, 1, &dims, NULL, copy,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
        PyArray_ENABLEFLAGS((PyArrayObject*) obj, NPY_ARRAY_OWNDATA);
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
    PyArrayObject* array = make_contiguous(obj, &is_new_object, 2,typecode, true);
    if (!array)
        return false;

    sg_matrix = shogun::SGMatrix<type>((type*) PyArray_DATA(array),
            PyArray_DIM(array,0), PyArray_DIM(array,1), true);

    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA);
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
            descr, 2, dims, NULL, (void*) copy,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
        PyArray_ENABLEFLAGS((PyArrayObject*) obj, NPY_ARRAY_OWNDATA);
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
    PyArrayObject* array = make_contiguous(obj, &is_new_object, -1,typecode, true);
    if (!array)
        return false;

    index_t ndim = PyArray_NDIM(array);
    if (ndim <= 0)
      return false;

    index_t* temp_dims = SG_MALLOC(index_t, ndim);

    npy_intp* py_dims = PyArray_DIMS(array);

    for (auto i=0; i<ndim; ++i)
      temp_dims[i] = py_dims[i];

    sg_array = SGNDArray<type>((type*) PyArray_DATA(array), temp_dims, ndim);

    PyArray_CLEARFLAGS(array, NPY_ARRAY_OWNDATA);
    Py_DECREF(array);

    return true;
}

template <class type>
static bool array_to_numpy(PyObject* &obj, SGNDArray<type> sg_array, int typecode)
{
	int n = 1;
#ifdef _MSC_VER
    npy_intp* dims = new npy_intp[sg_array.num_dims];
#else
	npy_intp dims[sg_array.num_dims];
#endif
	for (int i = 0; i < sg_array.num_dims; ++i)
	{
		dims[i] = (npy_intp)sg_array.dims[i];
		n *= sg_array.dims[i];
	}

	PyArray_Descr* descr=PyArray_DescrFromType(typecode);

	if (descr)
	{
		void* copy=get_copy(sg_array.array, sizeof(type)*size_t(n));
		obj = PyArray_NewFromDescr(&PyArray_Type,
		    descr, sg_array.num_dims, dims, NULL, (void*) copy,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
		PyArray_ENABLEFLAGS((PyArrayObject*) obj, NPY_ARRAY_OWNDATA);
	}
#ifdef _MSC_VER
    delete[] dims;
#endif

	return descr!=NULL;
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
    index_t num_feat=PyLong_AsLong(PyTuple_GetItem(shape, 0));
    index_t num_vec=PyLong_AsLong(PyTuple_GetItem(shape, 1));

    /* get indptr array */
    int is_new_object_indptr=0;
    PyArrayObject* array_indptr = make_contiguous(indptr, &is_new_object_indptr, 1, NPY_INT32);
    if (!array_indptr) return false;
    int32_t* bytes_indptr=(int32_t*) PyArray_DATA(array_indptr);
    int32_t len_indptr = PyArray_DIM(array_indptr,0);

    /* get indices array */
    int is_new_object_indices=0;
    PyArrayObject* array_indices = make_contiguous(indices, &is_new_object_indices, 1, NPY_INT32);
    if (!array_indices) return false;
    int32_t* bytes_indices=(int32_t*) PyArray_DATA(array_indices);
    int32_t len_indices = PyArray_DIM(array_indices,0);

    /* get data array */
    int is_new_object_data=0;
    PyArrayObject* array_data = make_contiguous(data, &is_new_object_data, 1, typecode);
    if (!array_data) return false;
    type* bytes_data=(type*) PyArray_DATA(array_data);
    int32_t len_data = PyArray_DIM(array_data,0);

    if (len_indices!=len_data)
        return false;

    shogun::SGSparseVector<type>* sfm = SG_MALLOC(shogun::SGSparseVector<type>, num_vec);

    for (auto i=1; i<len_indptr; ++i)
    {
        int32_t num = bytes_indptr[i]-bytes_indptr[i-1];

        if (num>0)
        {
            sfm[i-1]=SGSparseVector<type>(num);

            for (auto j=0; j<num; ++j)
            {
                sfm[i-1].features[j].feat_index=*bytes_indices;
                sfm[i-1].features[j].entry=*bytes_data;

                ++bytes_indices;
                ++bytes_data;
            }
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
    auto num_feat=sg_matrix.num_features;
    auto num_vec=sg_matrix.num_vectors;

    int64_t nnz=0;
    for (auto i=0; i<num_vec; ++i)
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
        index_t* indices = SG_MALLOC(index_t, nnz);
        type* data = SG_MALLOC(type, nnz);

        if (descr && descr_data && indptr && indices && data)
        {
            indptr[0]=0;

            index_t* i_ptr=indices;
            type* d_ptr=data;

            for (auto i=0; i<num_vec; ++i)
            {
                indptr[i+1]=indptr[i];
                indptr[i+1]+=sfm[i].num_feat_entries;

                for (auto j=0; j<sfm[i].num_feat_entries; ++j)
                {
                    *i_ptr=sfm[i].features[j].feat_index;
                    *d_ptr=sfm[i].features[j].entry;

                    ++i_ptr;
                    ++d_ptr;
                }
            }

            npy_intp indptr_dims = num_vec+1;
            indptr_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &indptr_dims, NULL, (void*) indptr,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
            PyArray_ENABLEFLAGS((PyArrayObject*) indptr_py, NPY_ARRAY_OWNDATA);

            npy_intp dims = nnz;
            indices_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr, 1, &dims, NULL, (void*) indices,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
            PyArray_ENABLEFLAGS((PyArrayObject*) indices_py, NPY_ARRAY_OWNDATA);

            data_py = PyArray_NewFromDescr(&PyArray_Type,
                    descr_data, 1, &dims, NULL, (void*) data,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
            PyArray_ENABLEFLAGS((PyArrayObject*) data_py, NPY_ARRAY_OWNDATA);

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

    index_t* indices = SG_MALLOC(index_t, dims);
    type* data = SG_MALLOC(type, dims);

    if (!(descr && descr_data && indices && data))
        return false;

    index_t* i_ptr=indices;
    type* d_ptr=data;

    for (auto j=0; j<sg_vector.num_feat_entries; ++j)
    {
        *i_ptr=sg_vector.features[j].feat_index;
        *d_ptr=sg_vector.features[j].entry;

        ++i_ptr;
        ++d_ptr;
    }

    indices_py = PyArray_NewFromDescr(&PyArray_Type,
            descr, 1, &dims, NULL, (void*) indices,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);

    PyArray_ENABLEFLAGS((PyArrayObject*) indices_py, NPY_ARRAY_OWNDATA);

    data_py = PyArray_NewFromDescr(&PyArray_Type,
            descr_data, 1, &dims, NULL, (void*) data,  NPY_ARRAY_FARRAY | NPY_ARRAY_WRITEABLE, NULL);
    PyArray_ENABLEFLAGS((PyArrayObject*) data_py, NPY_ARRAY_OWNDATA);

    PyTuple_SetItem(tuple, 0, data_py);
    PyTuple_SetItem(tuple, 1, indices_py);
    obj = tuple;
    return true;
}

%}

/* Features to ... */
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
		obj=SWIG_NewPointerObj(f, $descriptor(shogun::Features*), SWIG_POINTER_EXCEPTION);
		break;
	}
%enddef

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
%define TYPEMAP_SGVECTOR(type, typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGVector<type>, const shogun::SGVector<type>&
{
    $1 = is_pyvector($input, typecode);
}

// used for "in" typemap
%fragment(SWIG_AsVal_frag(shogun::SGVector<type>), "header")
{
    int SWIG_AsVal_dec(shogun::SGVector<type>)(PyObject* input, shogun::SGVector<type>& val)
    {
        return vector_from_numpy<type>(val, input, typecode)? SWIG_OK : SWIG_ERROR;
    }
}

// used for "out" typemap
%fragment(SWIG_From_frag(shogun::SGVector<type>), "header")
{
    PyObject* SWIG_From_dec(shogun::SGVector<type>)(const shogun::SGVector<type>& arg)
    {
        PyObject* result = nullptr;
        vector_to_numpy(result, arg, typecode);
        return result;
    }
}

// tell swig about the traits of the type to be used in stl containers
%fragment(SWIG_Traits_frag(shogun::SGVector<type>), "header",
          fragment="StdTraits",
          fragment=SWIG_From_frag(shogun::SGVector<type>),
          fragment=SWIG_AsVal_frag(shogun::SGVector<type>),
          fragment=SWIG_From_frag(std::string),
          fragment=SWIG_AsVal_frag(std::string))
{
namespace swig
{
    template <>
    struct traits<shogun::SGVector<type>>
    {
        typedef value_category category;
        static const char* type_name()
        {
            return "shogun::SGVector<" #type ">";
        }
    };

    template <>
    struct traits_asval<shogun::SGVector<type>>
    {
        typedef shogun::SGVector<type> value_type;

        static int asval(SWIG_Object obj, value_type *val)
        {
            if constexpr (!std::is_same<type, char>::value)
            {
                // do typechecking if val == nullptr
                if(!val)
                    return is_pyvector(obj, typecode)? SWIG_OK : SWIG_ERROR;
                else
                    return SWIG_AsVal_dec(shogun::SGVector<type>)(obj, *val);
            }
            else
            {
                // treat SGVector<char> as a string
                std::string str;
                int result = SWIG_AsVal_dec(std::string)(obj, &str);
                if (SWIG_IsOK(result) && val)
                {
                    const char* cstr = str.c_str();
                    *val = shogun::SGVector<type>(cstr, cstr + str.length());
                }
                return result;
            }
        }
    };

    template <>
    struct traits_from<shogun::SGVector<type>>
    {
        typedef shogun::SGVector<type> value_type;

        static SWIG_Object from(const value_type& val)
        {
            if constexpr (!std::is_same<type, char>::value)
            {
                return SWIG_From_dec(shogun::SGVector<type>)(val);
            }
            else
            {
                // treat SGVector<char> as a string
                std::stringstream ss;
                ss.write((char *)val.vector, val.vlen);
                return SWIG_From_dec(std::string)(ss.str());
            }
        }
    };
}
}

%val_in_typemap(shogun::SGVector<type>);
%val_out_typemap(shogun::SGVector<type>);

// vector array
%template() std::vector<shogun::SGVector<type>>;

%enddef

/* Define concrete examples of the TYPEMAP_SGVECTOR macros */
TYPEMAP_SGVECTOR(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_SGVECTOR(char,          NPY_UNICODE)
#else
TYPEMAP_SGVECTOR(char,          NPY_STRING)
#endif
TYPEMAP_SGVECTOR(uint8_t,       NPY_UINT8)
TYPEMAP_SGVECTOR(int16_t,       NPY_INT16)
TYPEMAP_SGVECTOR(uint16_t,      NPY_UINT16)
TYPEMAP_SGVECTOR(int32_t,       NPY_INT32)
TYPEMAP_SGVECTOR(uint32_t,      NPY_UINT32)
TYPEMAP_SGVECTOR(int64_t,       NPY_INT64)
TYPEMAP_SGVECTOR(uint64_t,      NPY_UINT64)
TYPEMAP_SGVECTOR(float32_t,     NPY_FLOAT32)
TYPEMAP_SGVECTOR(float64_t,     NPY_FLOAT64)
TYPEMAP_SGVECTOR(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SGVECTOR(complex128_t,   NPY_CDOUBLE)
TYPEMAP_SGVECTOR(PyObject,      NPY_OBJECT)

#undef TYPEMAP_SGVECTOR

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
TYPEMAP_IN_SGMATRIX(complex128_t,   NPY_CDOUBLE)
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
TYPEMAP_OUT_SGMATRIX(complex128_t,   NPY_CDOUBLE)
TYPEMAP_OUT_SGMATRIX(PyObject,      NPY_OBJECT)

#undef TYPEMAP_OUT_SGMATRIX

/* N-dimensional input arrays */
%define TYPEMAP_INND(type,typecode)
%typemap(typecheck, precedence=SWIG_TYPECHECK_POINTER) shogun::SGNDArray<type>
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

/* N-dimensional output arrays */
%define TYPEMAP_OUTND(type,typecode)
%typemap(out) shogun::SGNDArray<type>
{
    if (!array_to_numpy($result, $1, typecode))
        SWIG_fail;
}
%enddef

/* Define concrete examples of the TYPEMAP_OUTND macros */
TYPEMAP_OUTND(bool,          NPY_BOOL)
#ifdef PYTHON3 // str -> unicode for python3
TYPEMAP_OUTND(char,          NPY_UNICODE)
#else
TYPEMAP_OUTND(char,          NPY_STRING)
#endif
TYPEMAP_OUTND(uint8_t,       NPY_UINT8)
TYPEMAP_OUTND(int16_t,       NPY_INT16)
TYPEMAP_OUTND(uint16_t,      NPY_UINT16)
TYPEMAP_OUTND(int32_t,       NPY_INT32)
TYPEMAP_OUTND(uint32_t,      NPY_UINT32)
TYPEMAP_OUTND(int64_t,       NPY_INT64)
TYPEMAP_OUTND(uint64_t,      NPY_UINT64)
TYPEMAP_OUTND(float32_t,     NPY_FLOAT32)
TYPEMAP_OUTND(float64_t,     NPY_FLOAT64)
TYPEMAP_OUTND(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_OUTND(PyObject,      NPY_OBJECT)

#undef TYPEMAP_OUTND

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
TYPEMAP_SPARSEFEATURES_IN(complex128_t,  NPY_CDOUBLE)
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
TYPEMAP_SPARSEFEATURES_OUT(complex128_t,  NPY_CDOUBLE)
TYPEMAP_SPARSEFEATURES_OUT(floatmax_t,    NPY_LONGDOUBLE)
TYPEMAP_SPARSEFEATURES_OUT(PyObject,      NPY_OBJECT)
#undef TYPEMAP_SPARSEFEATURES_OUT

%typemap(throws) shogun::ShogunException
{
    PyErr_SetString(PyExc_SystemError, $1.what());
    SWIG_fail;
}

%feature("nothread") _rename_python_function;
%feature("docstring", "Renames a Python function in the given module or class. \n"
					  "Similar functionality to SWIG's %rename.") _rename_python_function;

%typemap(out) void _rename_python_function "$result = PyErr_Occurred() ? NULL : SWIG_Py_Void();"
%inline %{
	static void _rename_python_function(PyObject *type, PyObject *old_name, PyObject *new_name) {
		PyObject *dict = NULL,
				 *func_obj = NULL;
		if (!PyUnicode_Check(old_name) || !PyUnicode_Check(new_name))
			{
				PyErr_SetString(PyExc_TypeError, "'old_name' and 'new_name' have to be strings");
				return;
			}
		if (PyType_Check(type)) {
			PyTypeObject *pytype = (PyTypeObject *)type;
			dict = pytype->tp_dict;
			func_obj = PyDict_GetItem(dict, old_name);
			if (func_obj == NULL) {
				PyErr_SetString(PyExc_ValueError, "'old_name' name does not exist in the given type");
				return;
			}
		}
		else if ( PyModule_Check(type)) {
			dict = PyModule_GetDict(type);
			func_obj = PyDict_GetItem(dict, old_name);
			if (func_obj == NULL) {
				PyErr_SetString(PyExc_ValueError, "'old_name' does not exist in the given module");
				return;
			}
		}
		else {
			PyErr_SetString(PyExc_ValueError, "'type' is neither a module or a Python type");
			return;
		}
		if (PyDict_Contains(dict, new_name))
		{
			PyErr_SetString(PyExc_ValueError, "new_name already exists in the given scope");
			return;
		}
		PyDict_SetItem(dict, new_name, func_obj);
		PyDict_DelItem(dict, old_name);
  }
%}

#if SWIG_VERSION >= 0x04000
%pythoncode %{
  _sg_object = SGObject
  __version__ = Version_get_version_main()
%}
#else
%pythoncode %{
  _sg_object = _shogun.SGObject
  __version__ = _shogun.Version_get_version_main()
%}
#endif

%pythoncode %{
import sys

_FACTORIES = ["distance",
              "evaluation",
              "kernel",
              "machine",
              "multiclass_strategy",
              "ecoc_encoder",
              "ecoc_decoder",
              "transformer",
              "layer",
              "splitting_strategy",
              "machine_evaluation",
              "features",
              "differentiable",
              "gp_inference",
              "gp_mean",
              "gp_likelihood",
              "loss",
     ]

def _internal_factory_wrapper(object_name, new_name, docstring=None):
    """
    A wrapper that returns a generic factory that
    accepts kwargs and passes them to shogun.object_name
    via .put
    """
    _obj = getattr(sys.modules[__name__], object_name)
    def _internal_factory(name, *args, **kwargs):

        new_obj = _obj(name, *args)
        for k,v in kwargs.items():
            new_obj.put(k, v)
        return new_obj
    if docstring:
        _internal_factory.__doc__ = docstring
    else:
        _internal_factory.__doc__ = _obj.__doc__.replace(object_name, new_name)
    _internal_factory.__qualname__ = new_name

    return _internal_factory

for factory in _FACTORIES:
    # renames function in the current module (shogun) from `factory` to "_" + `factory`
    # which "hides" it from the user
    factory_private_name = "_{}".format(factory)
    _rename_python_function(sys.modules[__name__], factory, factory_private_name)
    # adds a new function called `factory` to the shogun module which is a wrapper
    # that passes **kwargs to objects via .put (see _internal_factory_wrapper)
    _swig_monkey_patch(sys.modules[__name__], factory, _internal_factory_wrapper(factory_private_name, factory))

%}

#endif /* HAVE_PYTHON */
