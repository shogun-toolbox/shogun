#include "PythonInterface.h"

#include <stdio.h>
#include <dlfcn.h>

#include <shogun/lib/ShogunException.h>
#include <shogun/io/SGIO.h>
#include <shogun/ui/SGInterface.h>
#include <shogun/base/init.h>

#ifdef HAVE_OCTAVE
#include "../octave_static/OctaveInterface.h"
#endif

#ifdef HAVE_R
#include "../r_static/RInterface.h"
#endif

using namespace shogun;

void* CPythonInterface::m_pylib=0;

void python_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void python_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		PyErr_Warn(NULL, (char*) str); //the cast seems to be necessary for python 2.4.3
	else
		fprintf(target, "%s", str);
}

void python_print_error(FILE* target, const char* str)
{
	if (target==stdout)
		PyErr_SetString(PyExc_RuntimeError, str);
	else
		fprintf(target, "%s", str);
}

void python_cancel_computations(bool &delayed, bool &immediately)
{
	if (PyErr_CheckSignals())
	{
		SG_SPRINT("\nImmediately return to matlab prompt / Prematurely finish computations / Do nothing (I/P/D)? ");
		char answer=fgetc(stdin);

		if (answer == 'I')
			immediately=true;
		else if (answer == 'P')
		{
			PyErr_Clear();
			delayed=true;
		}
		else
			SG_SPRINT("\n");
	}
}

extern CSGInterface* interface;

CPythonInterface::CPythonInterface(PyObject* args)
: CSGInterface(false)
{
	reset(NULL, args);
}

CPythonInterface::CPythonInterface(PyObject* self, PyObject* args)
: CSGInterface()
{
	reset(self, args);
}

CPythonInterface::~CPythonInterface()
{
}

void CPythonInterface::reset(PyObject* self, PyObject* args)
{
	CSGInterface::reset();

	ASSERT(PyTuple_Check(args));
	m_rhs=args;
	m_nrhs=PyTuple_GET_SIZE(args);

	m_nlhs=0;
	Py_INCREF(Py_None);
	m_lhs=Py_None;
}


/** get functions - to pass data from the target interface to shogun */


/// get type of current argument (does not increment argument counter)
IFType CPythonInterface::get_argument_type()
{
	PyObject* arg= PyTuple_GetItem(m_rhs, m_rhs_counter);
	ASSERT(arg);

	if (PyList_Check(arg) && PyList_Size((PyObject *) arg)>0)
	{
		PyObject* item= PyList_GetItem((PyObject *) arg, 0);

#ifdef IS_PYTHON3
		if (PyUnicode_Check(item))
#else
		if (PyString_Check(item))
#endif
		{
			return STRING_CHAR;
		}
	}
	else if PyArray_Check(arg)
	{
		if (PyArray_TYPE(arg)==NPY_CHAR)
			return STRING_CHAR;
		if (PyArray_TYPE(arg)==NPY_BYTE)
			return STRING_BYTE;
		if (PyArray_TYPE(arg)==NPY_INT)
			return DENSE_INT;
		if (PyArray_TYPE(arg)==NPY_DOUBLE)
			return DENSE_REAL;
		if (PyArray_TYPE(arg)==NPY_SHORT)
			return DENSE_SHORT;
		if (PyArray_TYPE(arg)==NPY_FLOAT)
			return DENSE_SHORTREAL;
		if (PyArray_TYPE(arg)==NPY_USHORT)
			return DENSE_WORD;
	}
	return UNDEFINED;
}


int32_t CPythonInterface::get_int()
{
	const PyObject* i = get_arg_increment();
	if (!i || !PyInt_Check(i))
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

    return PyInt_AS_LONG(const_cast<PyObject*>(i));
}

float64_t CPythonInterface::get_real()
{
	const PyObject* f=get_arg_increment();
	if (!f || !PyFloat_Check(f))
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return PyFloat_AS_DOUBLE(f);
}

bool CPythonInterface::get_bool()
{
	const PyObject* b=get_arg_increment();
	if (!b || !PyBool_Check(b))
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

    return PyInt_AS_LONG(const_cast<PyObject*>(b)) != 0;
}

char* CPythonInterface::get_string(int32_t& len)
{
	const PyObject* s=get_arg_increment();

#ifdef IS_PYTHON3
    if (!s || !PyUnicode_Check(s))
        SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

    len = PyUnicode_GetSize((PyObject*) s);
    char* str = PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<PyObject*>(s)));
#else
    if (!s || !PyString_Check(s))
        SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

    len = PyString_Size((PyObject*) s);
	char* str = PyString_AS_STRING(s);
	ASSERT(str && len>0);
#endif

    ASSERT(str && len>0);

	char* cstr=SG_MALLOC(char, len+1);
	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	return cstr;
}

#define GET_VECTOR(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(sg_type*& vector, int32_t& len)		\
{																			\
	const PyArrayObject* py_vec=(PyArrayObject *) get_arg_increment();		\
	if (!py_vec || !PyArray_Check(py_vec) || py_vec->nd!=1 ||				\
			PyArray_TYPE(py_vec)!=py_type)									\
	{																		\
		SG_ERROR("Expected " error_string " Vector as argument %d\n",		\
			m_rhs_counter);												\
	}																		\
																			\
	len=py_vec->dimensions[0];												\
	npy_intp stride_offs= py_vec->strides[0];								\
	vector=SG_MALLOC(sg_type, len);												\
	char* data=(char*) py_vec->data;										\
	npy_intp offs=0;														\
																			\
	for (int32_t i=0; i<len; i++)											\
	{																		\
		vector[i]=*((if_type*)(data+offs));									\
		offs+=stride_offs;													\
	}																		\
}

GET_VECTOR(get_vector, NPY_BYTE, uint8_t, uint8_t, "Byte")
GET_VECTOR(get_vector, NPY_CHAR, char, char, "Char")
GET_VECTOR(get_vector, NPY_INT, int32_t, int, "Integer")
GET_VECTOR(get_vector, NPY_SHORT, int16_t, short, "Short")
GET_VECTOR(get_vector, NPY_FLOAT, float32_t, float, "Single Precision")
GET_VECTOR(get_vector, NPY_DOUBLE, float64_t, double, "Double Precision")
GET_VECTOR(get_vector, NPY_USHORT, uint16_t, unsigned short, "Word")
#undef GET_VECTOR


#define GET_MATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																			\
	const PyArrayObject* py_mat=(PyArrayObject *) get_arg_increment();		\
	if (!py_mat || !PyArray_Check(py_mat) ||								\
			PyArray_TYPE(py_mat)!=py_type || py_mat->nd!=2)				\
	{																		\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n",		\
			m_rhs_counter);												\
	}																		\
																			\
	num_feat=py_mat->dimensions[0];										\
	num_vec=py_mat->dimensions[1];											\
	matrix=SG_MALLOC(sg_type, num_vec*num_feat);									\
	char* data=py_mat->data;												\
	npy_intp* strides= py_mat->strides;									\
	npy_intp d2_offs=0;														\
	for (int32_t i=0; i<num_feat; i++)										\
	{																		\
		npy_intp offs=d2_offs;												\
		for (int32_t j=0; j<num_vec; j++)									\
		{																	\
			matrix[i+j*num_feat]=*((if_type*)(data+offs));					\
			offs+=strides[1];												\
		}																	\
		d2_offs+=strides[0];												\
	}																		\
}

GET_MATRIX(get_matrix, NPY_BYTE, uint8_t, uint8_t, "Byte")
GET_MATRIX(get_matrix, NPY_CHAR, char, char, "Char")
GET_MATRIX(get_matrix, NPY_INT, int32_t, int, "Integer")
GET_MATRIX(get_matrix, NPY_SHORT, int16_t, short, "Short")
GET_MATRIX(get_matrix, NPY_FLOAT, float32_t, float, "Single Precision")
GET_MATRIX(get_matrix, NPY_DOUBLE, float64_t, double, "Double Precision")
GET_MATRIX(get_matrix, NPY_USHORT, uint16_t, unsigned short, "Word")
#undef GET_MATRIX

#define GET_NDARRAY(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(sg_type*& array, int32_t*& dims, int32_t& num_dims)	\
{																			\
	const PyArrayObject* py_mat=(PyArrayObject *) get_arg_increment();		\
	if (!py_mat || !PyArray_Check(py_mat) ||								\
			PyArray_TYPE(py_mat)!=py_type)									\
	{																		\
		SG_ERROR("Expected " error_string " ND-Array as argument %d\n",		\
			m_rhs_counter);												\
	}																		\
																			\
	num_dims=py_mat->nd;													\
	int64_t total_size=0;														\
																			\
	dims=SG_MALLOC(int32_t, num_dims);													\
	for (int32_t d=0; d<num_dims; d++)											\
	{																		\
		dims[d]=(int32_t) py_mat->dimensions[d];								\
		total_size+=dims[d];												\
	}																		\
																			\
	array=SG_MALLOC(sg_type, total_size);											\
																			\
	char* data=py_mat->data;												\
	for (int64_t i=0; i<total_size; i++)										\
		array[i]=*(((if_type*)(data))+i);									\
}

GET_NDARRAY(get_ndarray, NPY_BYTE, uint8_t, uint8_t, "Byte")
GET_NDARRAY(get_ndarray, NPY_CHAR, char, char, "Char")
GET_NDARRAY(get_ndarray, NPY_INT, int32_t, int, "Integer")
GET_NDARRAY(get_ndarray, NPY_SHORT, int16_t, short, "Short")
GET_NDARRAY(get_ndarray, NPY_FLOAT, float32_t, float, "Single Precision")
GET_NDARRAY(get_ndarray, NPY_DOUBLE, float64_t, double, "Double Precision")
GET_NDARRAY(get_ndarray, NPY_USHORT, uint16_t, unsigned short, "Word")
#undef GET_NDARRAY


#define GET_SPARSEMATRIX(function_name, py_type, sg_type, if_type, error_string) \
void CPythonInterface::function_name(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec) \
{																			\
	/* no sparse available yet */ \
	return; \
	\
	/* \
	const PyArray_Object* py_mat=(PyArrayObject *) get_arg_increment();	\
	if (!PyArray_Check(py_mat))											\
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter); \
																			\
	if (!PyArray_TYPE(py_mat)!=py_type)									\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n",		\
			m_rhs_counter);												\
																			\
	num_vec=py_mat->dimensions[0];											\
	num_feat=py_mat->nd;													\
	matrix=SG_MALLOC(SGSparseVector<sg_type>, num_vec);									\
	if_type* data=(if_type*) py_mat->data;									\
																			\
	int64_t nzmax=mxGetNzmax(mx_mat);											\
	mwIndex* ir=mxGetIr(mx_mat);											\
	mwIndex* jc=mxGetJc(mx_mat);											\
	int64_t offset=0;															\
	for (int32_t i=0; i<num_vec; i++)											\
	{																		\
		int32_t len=jc[i+1]-jc[i];												\
		matrix[i].num_feat_entries=len;										\
																			\
		if (len>0)															\
		{																	\
			matrix[i].features=SG_MALLOC(SGSparseVectorEntry<sg_type>, len);				\
			for (int32_t j=0; j<len; j++)										\
			{																\
				matrix[i].features[j].entry=data[offset];					\
				matrix[i].features[j].feat_index=ir[offset];				\
				offset++;													\
			}																\
		}																	\
		else																\
			matrix[i].features=NULL;										\
	}																		\
	ASSERT(offset==nzmax);													\
	*/ \
}

GET_SPARSEMATRIX(get_sparse_matrix, NPY_DOUBLE, float64_t, double, "Double Precision")
/*  future versions might support types other than float64_t
GET_SPARSEMATRIX(get_sparse_matrix, "uint8", uint8_t, uint8_t, "Byte")
GET_SPARSEMATRIX(get_sparse_matrix, "char", char, mxChar, "Char")
GET_SPARSEMATRIX(get_sparse_matrix, "int32", int32_t, int, "Integer")
GET_SPARSEMATRIX(get_sparse_matrix, "int16", int16_t, short, "Short")
GET_SPARSEMATRIX(get_sparse_matrix, "single", float32_t, float, "Single Precision")
GET_SPARSEMATRIX(get_sparse_matrix, "uint16", uint16_t, unsigned short, "Word")*/
#undef GET_SPARSEMATRIX

#ifdef IS_PYTHON3

#define GET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len)	\
{																			\
	max_string_len=0;														\
	const PyObject* py_str= get_arg_increment();							\
	if (!py_str)															\
		SG_ERROR("Expected Stringlist as argument (none given).\n");		\
																			\
	if (PyList_Check(py_str))												\
	{																		\
		if (!is_char_str)													\
			SG_ERROR("Only Character Strings supported.\n");				\
																			\
		num_str=PyList_Size((PyObject*) py_str);							\
		ASSERT(num_str>=1);													\
																			\
		strings=SG_MALLOC(SGString<sg_type>, num_str);								\
		ASSERT(strings);													\
																			\
		for (int32_t i=0; i<num_str; i++)										\
		{																	\
			PyObject *o = PyList_GetItem((PyObject*) py_str,i);				\
			if (PyUnicode_Check(o))											\
			{																\
				int32_t len = PyUnicode_GetSize((PyObject*) o);									\
				const sg_type* str = (const sg_type*) PyBytes_AsString(PyUnicode_AsASCIIString(const_cast<PyObject*>(o))); \
																			\
				strings[i].slen=len;										\
				strings[i].string=NULL;										\
				max_string_len=CMath::max(max_string_len, len);				\
																			\
				if (len>0)													\
				{															\
					strings[i].string=SG_MALLOC(sg_type, len+1);					\
					memcpy(strings[i].string, str, len);					\
					strings[i].string[len]='\0';							\
				}															\
			}																\
			else															\
			{																\
				for (int32_t j=0; j<i; j++)										\
					SG_FREE(strings[i].string);								\
				SG_FREE(strings);											\
				SG_ERROR("All elements in list must be strings, error in line %d.\n", i);\
			}																\
		}																	\
	}																		\
	else if (PyArray_TYPE(py_str)==py_type && ((PyArrayObject*) py_str)->nd==2)	\
	{																		\
		const PyArrayObject* py_array_str=(const PyArrayObject*) py_str;	\
		if_type* data=(if_type*) py_array_str->data;						\
		num_str=py_array_str->dimensions[0];								\
		int32_t len=py_array_str->dimensions[1];								\
		strings=SG_MALLOC(SGString<sg_type>, num_str);							\
																			\
		for (int32_t i=0; i<num_str; i++)										\
		{																	\
			if (len>0)														\
			{																\
				strings[i].slen=len; /* all must have same length*/			\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated */	\
				int32_t j;														\
				for (j=0; j<len; j++)										\
					strings[i].string[j]=data[j+i*len];					\
				strings[i].string[j]='\0';									\
			}																\
			else															\
			{																\
				SG_WARNING( "string with index %d has zero length.\n", i+1);	\
				strings[i].slen=0;											\
				strings[i].string=NULL;									\
			}																\
		}																	\
		max_string_len=len;													\
	}																		\
	else																	\
		SG_ERROR("Expected String as argument %d.\n", m_rhs_counter);		\
}

#else

#define GET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len)	\
{																			\
	max_string_len=0;														\
	const PyObject* py_str= get_arg_increment();							\
	if (!py_str)															\
		SG_ERROR("Expected Stringlist as argument (none given).\n");		\
																			\
	if (PyList_Check(py_str))												\
	{																		\
		if (!is_char_str)													\
			SG_ERROR("Only Character Strings supported.\n");				\
																			\
		num_str=PyList_Size((PyObject*) py_str);							\
		ASSERT(num_str>=1);													\
																			\
		strings=SG_MALLOC(SGString<sg_type>, num_str);								\
		ASSERT(strings);													\
																			\
		for (int32_t i=0; i<num_str; i++)										\
		{																	\
			PyObject *o = PyList_GetItem((PyObject*) py_str,i);				\
			if (PyString_Check(o))											\
			{																\
				int32_t len=PyString_Size(o);									\
				const sg_type* str= (const sg_type*) PyString_AsString(o);	\
																			\
				strings[i].slen=len;										\
				strings[i].string=NULL;										\
				max_string_len=CMath::max(max_string_len, len);				\
																			\
				if (len>0)													\
				{															\
					strings[i].string=SG_MALLOC(sg_type, len+1);					\
					memcpy(strings[i].string, str, len);					\
					strings[i].string[len]='\0';							\
				}															\
			}																\
			else															\
			{																\
				for (int32_t j=0; j<i; j++)										\
					SG_FREE(strings[i].string);								\
				SG_FREE(strings);											\
				SG_ERROR("All elements in list must be strings, error in line %d.\n", i);\
			}																\
		}																	\
	}																		\
	else if (PyArray_TYPE(py_str)==py_type && ((PyArrayObject*) py_str)->nd==2)	\
	{																		\
		const PyArrayObject* py_array_str=(const PyArrayObject*) py_str;	\
		if_type* data=(if_type*) py_array_str->data;						\
		num_str=py_array_str->dimensions[0];								\
		int32_t len=py_array_str->dimensions[1];								\
		strings=SG_MALLOC(SGString<sg_type>, num_str);							\
																			\
		for (int32_t i=0; i<num_str; i++)										\
		{																	\
			if (len>0)														\
			{																\
				strings[i].slen=len; /* all must have same length*/			\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated */	\
				int32_t j;														\
				for (j=0; j<len; j++)										\
					strings[i].string[j]=data[j+i*len];					\
				strings[i].string[j]='\0';									\
			}																\
			else															\
			{																\
				SG_WARNING( "string with index %d has zero length.\n", i+1);	\
				strings[i].slen=0;											\
				strings[i].string=NULL;									\
			}																\
		}																	\
		max_string_len=len;													\
	}																		\
	else																	\
		SG_ERROR("Expected String as argument %d.\n", m_rhs_counter);		\
}

#endif

GET_STRINGLIST(get_string_list, NPY_BYTE, uint8_t, uint8_t, 1, "Byte")
GET_STRINGLIST(get_string_list, NPY_CHAR, char, char, 1, "Char")
GET_STRINGLIST(get_string_list, NPY_INT, int32_t, int, 0, "Integer")
GET_STRINGLIST(get_string_list, NPY_SHORT, int16_t, short, 0, "Short")
GET_STRINGLIST(get_string_list, NPY_USHORT, uint16_t, unsigned short, 0, "Word")
#undef GET_STRINGLIST

void CPythonInterface::get_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* &attrs)
{
	attrs=NULL;
}


/** set functions - to pass data from shogun to the target interface */

void CPythonInterface::set_int(int32_t scalar)
{
	PyObject* o=Py_BuildValue("i", scalar);
	if (!o)
		SG_ERROR("Could not build an integer.\n");

	set_arg_increment(o);
}

void CPythonInterface::set_real(float64_t scalar)
{
	PyObject* o=Py_BuildValue("d", scalar);
	if (!o)
		SG_ERROR("Could not build a double.\n");

	set_arg_increment(o);
}

void CPythonInterface::set_bool(bool scalar)
{
	// bool does not exist in Py_BuildValue, using byte instead
	PyObject* o=Py_BuildValue("b", scalar);
	if (!o)
		SG_ERROR("Could not build a bool.\n");

	set_arg_increment(o);
}


#define SET_VECTOR(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const sg_type* vector, int32_t len)		\
{																			\
	if (!vector || len<1)													\
		SG_ERROR("Given vector is invalid.\n");								\
																			\
	npy_intp dims=len;														\
	PyObject* py_vec=PyArray_SimpleNew(1, &dims, py_type);					\
	if (!py_vec || !PyArray_Check(py_vec))									\
		SG_ERROR("Couldn't create " error_string " Vector of length %d.\n",	\
			len);															\
																			\
	if_type* data=(if_type*) ((PyArrayObject *) py_vec)->data;				\
																			\
	for (int32_t i=0; i<len; i++)												\
		data[i]=vector[i];													\
																			\
	set_arg_increment(py_vec);												\
}

SET_VECTOR(set_vector, NPY_BYTE, uint8_t, uint8_t, "Byte")
SET_VECTOR(set_vector, NPY_CHAR, char, char, "Char")
SET_VECTOR(set_vector, NPY_INT, int32_t, int, "Integer")
SET_VECTOR(set_vector, NPY_SHORT, int16_t, short, "Short")
SET_VECTOR(set_vector, NPY_FLOAT, float32_t, float, "Single Precision")
SET_VECTOR(set_vector, NPY_DOUBLE, float64_t, double, "Double Precision")
SET_VECTOR(set_vector, NPY_USHORT, uint16_t, unsigned short, "Word")
#undef SET_VECTOR


#define SET_MATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																			\
	if (!matrix || num_feat<1 || num_vec<1)								\
		SG_ERROR("Given matrix is invalid.\n");								\
																			\
	npy_intp dims[2]={num_feat, num_vec};									\
	PyObject* py_mat=PyArray_SimpleNew(2, dims, py_type);					\
	if (!py_mat || !PyArray_Check(py_mat))									\
		SG_ERROR("Couldn't create " error_string " Matrix of %d rows and %d cols.\n",	\
			num_feat, num_vec);												\
	ASSERT(PyArray_ISCARRAY(py_mat));										\
																			\
	if_type* data=(if_type*) ((PyArrayObject *) py_mat)->data;				\
																			\
	for (int32_t j=0; j<num_feat; j++)											\
		for (int32_t i=0; i<num_vec; i++)										\
			data[i+j*num_vec]=matrix[i*num_feat+j];						\
																			\
	set_arg_increment(py_mat);												\
}

SET_MATRIX(set_matrix, NPY_BYTE, uint8_t, uint8_t, "Byte")
SET_MATRIX(set_matrix, NPY_CHAR, char, char, "Char")
SET_MATRIX(set_matrix, NPY_INT, int32_t, int, "Integer")
SET_MATRIX(set_matrix, NPY_SHORT, int16_t, short, "Short")
SET_MATRIX(set_matrix, NPY_FLOAT, float32_t, float, "Single Precision")
SET_MATRIX(set_matrix, NPY_DOUBLE, float64_t, double, "Double Precision")
SET_MATRIX(set_matrix, NPY_USHORT, uint16_t, unsigned short, "Word")
#undef SET_MATRIX

#define SET_SPARSEMATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)	\
{																			\
	/* no sparse available yet */ \
	return; \
	/*\
	if (!matrix)															\
		SG_ERROR("Given matrix is invalid.\n");								\
																			\
	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, nnz, mxREAL);			\
	if (!mx_mat)															\
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec); \
																			\
	if_type* data=(if_type*) mxGetData(mx_mat);								\
																			\
	mwIndex* ir=mxGetIr(mx_mat);											\
	mwIndex* jc=mxGetJc(mx_mat);											\
	int64_t offset=0;															\
	for (int32_t i=0; i<num_vec; i++)											\
	{																		\
		int32_t len=matrix[i].num_feat_entries;									\
		jc[i]=offset;														\
		for (int32_t j=0; j<len; j++)											\
		{																	\
			data[offset]=matrix[i].features[j].entry;						\
			ir[offset]=matrix[i].features[j].feat_index;					\
			offset++;														\
		}																	\
	}																		\
	jc[num_vec]=offset;														\
																			\
	set_arg_increment(mx_mat);												\
	*/ \
}

SET_SPARSEMATRIX(set_sparse_matrix, NPY_DOUBLE, float64_t, double, "Double Precision")

/* future version might support this
SET_SPARSEMATRIX(set_sparse_matrix, mxUINT8_CLASS, uint8_t, uint8_t, "Byte")
SET_SPARSEMATRIX(set_sparse_matrix, mxCHAR_CLASS, char, mxChar, "Char")
SET_SPARSEMATRIX(set_sparse_matrix, mxINT32_CLASS, int32_t, int, "Integer")
SET_SPARSEMATRIX(set_sparse_matrix, mxINT16_CLASS, int16_t, short, "Short")
SET_SPARSEMATRIX(set_sparse_matrix, mxSINGLE_CLASS, float32_t, float, "Single Precision")
SET_SPARSEMATRIX(set_sparse_matrix, mxUINT16_CLASS, uint16_t, unsigned short, "Word")*/
#undef SET_SPARSEMATRIX

#ifdef IS_PYTHON3

#define SET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(const SGString<sg_type>* strings, int32_t num_str)	\
{																				\
	if (!is_char_str)															\
		SG_ERROR("Only character strings supported.\n");						\
																				\
	if (!strings || num_str<1)													\
		SG_ERROR("Given strings are invalid.\n");								\
																				\
	PyObject* py_str=PyList_New(num_str);										\
	if (!py_str || PyTuple_GET_SIZE(py_str)!=num_str)							\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str);		\
																				\
	for (int32_t i=0; i<num_str; i++)											\
	{																			\
		int32_t len=strings[i].slen;											\
		if (len>0)																\
		{																		\
			PyObject* str=PyUnicode_FromStringAndSize((const char*) strings[i].string, len); \
			if (!str)															\
				SG_ERROR("Couldn't create " error_string						\
						" String %d of length %d.\n", i, len);					\
																				\
			PyList_SET_ITEM(py_str, i, str);									\
		}																		\
	}																			\
																				\
	set_arg_increment(py_str);													\
}

#else

#define SET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(const SGString<sg_type>* strings, int32_t num_str)	\
{																				\
	if (!is_char_str)															\
		SG_ERROR("Only character strings supported.\n");						\
																				\
	if (!strings || num_str<1)													\
		SG_ERROR("Given strings are invalid.\n");								\
																				\
	PyObject* py_str=PyList_New(num_str);										\
	if (!py_str || PyTuple_GET_SIZE(py_str)!=num_str)							\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str);		\
																				\
	for (int32_t i=0; i<num_str; i++)											\
	{																			\
		int32_t len=strings[i].slen;											\
		if (len>0)																\
		{																		\
			PyObject* str=PyString_FromStringAndSize((const char*) strings[i].string, len); \
			if (!str)															\
				SG_ERROR("Couldn't create " error_string						\
						" String %d of length %d.\n", i, len);					\
																				\
			PyList_SET_ITEM(py_str, i, str);									\
		}																		\
	}																			\
																				\
	set_arg_increment(py_str);													\
}

#endif

SET_STRINGLIST(set_string_list, NPY_BYTE, uint8_t, uint8_t, 0, "Byte")
SET_STRINGLIST(set_string_list, NPY_CHAR, char, char, 1, "Char")
SET_STRINGLIST(set_string_list, NPY_INT, int32_t, int, 0, "Integer")
SET_STRINGLIST(set_string_list, NPY_SHORT, int16_t, short, 0, "Short")
SET_STRINGLIST(set_string_list, NPY_USHORT, uint16_t, unsigned short, 0, "Word")
#undef SET_STRINGLIST

void CPythonInterface::set_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* attrs)
{
}

bool CPythonInterface::create_return_values(int32_t num)
{
	if (num<=0)
		return true;

	m_lhs=PyTuple_New(num);
	ASSERT(m_lhs);

	m_nlhs=num;
	return PyTuple_GET_SIZE(m_lhs)==num;
}


void CPythonInterface::run_python_init()
{
#ifdef LIBPYTHON
	m_pylib = dlopen(LIBPYTHON, RTLD_NOW | RTLD_GLOBAL);
	if (!m_pylib)
		SG_SERROR("couldn't open " LIBPYTHON ".so\n");
#endif
	Py_Initialize();
    init_numpy();
}

void CPythonInterface::run_python_exit()
{
	Py_Finalize();
#ifdef LIBPYTHON
	dlclose(m_pylib);
#endif
}

bool CPythonInterface::run_python_helper(CSGInterface* from_if)
{
	SG_OBJ_DEBUG(from_if, "Entering Python\n")
	PyObject* globals = PyDict_New();
	PyObject* builtins = PyEval_GetBuiltins();
	PyDict_SetItemString(globals,"__builtins__", builtins);
	char* python_code=NULL;

	for (int i=0; i<from_if->get_nrhs(); i++)
	{
		int len=0;
		char* var_name = from_if->get_string(len);
		SG_OBJ_DEBUG(from_if, "var_name = '%s'\n", var_name);
		if (strmatch(var_name, "pythoncode"))
		{
			len=0;
			python_code=from_if->get_string(len);
			SG_OBJ_DEBUG(from_if, "python_code = '%s'\n", python_code);
			break;
		}
		else
		{
			PyObject* tuple = PyTuple_New(1);

			CPythonInterface* in = new CPythonInterface(tuple);
			in->create_return_values(1);
			from_if->translate_arg(from_if, in);
			PyDict_SetItemString(globals, var_name, in->get_return_values());
			SG_FREE(var_name);
			Py_DECREF(tuple);
			SG_UNREF(in);
		}
	}

	PyObject* python_code_obj = Py_CompileString(python_code, "<pythoncode>", Py_file_input);
	if (python_code_obj == NULL)
	{
		PyErr_Print();
		SG_OBJ_ERROR(from_if, "Compiling python code failed.");
	}

	SG_FREE(python_code);

#ifdef IS_PYTHON3
    PyObject* res = PyEval_EvalCode(python_code_obj, globals, NULL);
#else
    PyObject* res = PyEval_EvalCode((PyCodeObject*) python_code_obj, globals, NULL);
#endif

	Py_DECREF(python_code_obj);

	if (res == NULL)
	{
		PyErr_Print();
		SG_OBJ_ERROR(from_if, "Running python code failed.\n");
	}
	else
		SG_OBJ_DEBUG(from_if, "Successfully executed python code.\n");

	Py_DECREF(res);

	PyObject* results = PyDict_GetItemString(globals, "results");
	int32_t sz=-1;

	if (results)
	{
		if (!PyTuple_Check(results))
			SG_OBJ_ERROR(from_if, "results should be a tuple, e.g. results=(1,2,3) or results=tuple([42])")
		else
			sz=PyTuple_Size(results);
	}

	if (sz>0 && from_if->create_return_values(sz))
	{
		CPythonInterface* out = new CPythonInterface(results);

		//process d
		for (int32_t i=0; i<sz; i++)
			from_if->translate_arg(out, from_if);

		Py_DECREF(results);
		SG_UNREF(out);
	}
	else
	{
		if (sz>-1 && sz!=from_if->get_nlhs())
		{
			SG_OBJ_ERROR(from_if, "Number of return values (%d) does not match number of expected"
					" return values (%d).\n", sz, from_if->get_nlhs())
		}
	}

	Py_DECREF(globals);
	SG_OBJ_DEBUG(from_if, "Leaving Python.\n");
	return true;
}

bool CPythonInterface::cmd_run_octave()
{
#ifdef HAVE_OCTAVE
	return COctaveInterface::run_octave_helper(this);
#else
	return false;
#endif
}

bool CPythonInterface::cmd_run_r()
{
#ifdef HAVE_R
	return CRInterface::run_r_helper(this);
#else
	return false;
#endif
}


#ifdef HAVE_ELWMS
PyObject* elwms(PyObject* self, PyObject* args)
#else
PyObject* sg(PyObject* self, PyObject* args)
#endif
{
	try
	{
		if (!interface)
			interface=new CPythonInterface(self, args);
		else
			((CPythonInterface*) interface)->reset(self, args);

		if (!interface->handle())
			SG_SERROR("Unknown command.\n");
	}
	catch (std::bad_alloc)
	{
		SG_SPRINT("Out of memory error.\n");
		return NULL;
	}
	catch (ShogunException e)
	{
		PyErr_SetString(PyExc_RuntimeError, e.get_exception_string());
		return NULL;
	}
	catch (...)
	{
		return NULL;
	}

	return ((CPythonInterface*) interface)->get_return_values();
}

void exitsg(void)
{
	SG_SINFO("Quitting...\n");
#ifdef HAVE_OCTAVE
	COctaveInterface::run_octave_exit();
#endif
#ifdef HAVE_R
	CRInterface::run_r_exit();
#endif
	exit_shogun();
}

static PyMethodDef sg_pythonmethods[] = {
	{(char*)
#ifdef HAVE_ELWMS
	"elwms", elwms, METH_VARARGS, (char*) "Shogun."},
#else
    "sg", sg, METH_VARARGS, (char*) "Shogun."},
#endif
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

#ifdef HAVE_ELWMS
MOD_INIT(elwms)
#else
MOD_INIT(sg)
#endif
{
    PyObject *module;

	// initialize python interpreter
	Py_Initialize();

	// initialize threading (just in case it is needed)
	PyEval_InitThreads();

    // callback to cleanup at exit
	Py_AtExit(exitsg);

	// initialize callbacks
#ifdef HAVE_ELWMS
    MOD_DEF(module, (char*) "elwms", sg_pythonmethods);
#else
    MOD_DEF(module, (char*) "sg", sg_pythonmethods);
#endif

    if (module == NULL)
        return MOD_ERROR_VAL;

#ifdef HAVE_OCTAVE
	COctaveInterface::run_octave_init();
#endif
#ifdef HAVE_R
	CRInterface::run_r_init();
#endif
    init_numpy();

	// init_shogun has to be called before anything else
	// exit_shogun is called upon destruction in exitsg()
	init_shogun(&python_print_message, &python_print_warning,
			&python_print_error, &python_cancel_computations);

    return MOD_SUCCESS_VAL(module);
}
