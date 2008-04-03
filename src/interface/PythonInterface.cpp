#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)            

#include "interface/PythonInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/python.h"

extern "C" {
#include <object.h>
#include <../../numarray/numpy/libnumarray.h>
#include <numpy/ndarrayobject.h>
}

extern CSGInterface* interface;

CPythonInterface::CPythonInterface(PyObject* self, PyObject* args) : CSGInterface()
{
	ASSERT(PyTuple_Check(args));
	m_rhs=args;
	m_nrhs=PyTuple_GET_SIZE(args);

	m_nlhs=0;
	Py_INCREF(Py_None);
	m_lhs=Py_None;

	import_libnumarray();
	import_array();
}

CPythonInterface::~CPythonInterface()
{
}

/** get functions - to pass data from the target interface to shogun */
void CPythonInterface::parse_args(INT num_args, INT num_default_args)
{
}


/// get type of current argument (does not increment argument counter)
IFType CPythonInterface::get_argument_type()
{
	PyArrayObject* arg=(PyArrayObject*) PyTuple_GET_ITEM(m_rhs, m_rhs_counter);
	ASSERT(arg);

	if (!PyArray_Check(arg))
		return UNDEFINED;

	if (PyArray_TYPE(arg)==NPY_STRING)
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

	if (PyList_Check(arg) && PyList_Size((PyObject *) arg)>0)
	{
		PyArrayObject *item=
			(PyArrayObject *) PyList_GetItem((PyObject *) arg, 0);
		if (PyArray_Check(item) && item->nd==1)
		{
			if (PyArray_TYPE(item)==NPY_STRING)
				return STRING_CHAR;
			if (PyArray_TYPE(item)==NPY_BYTE)
				return STRING_BYTE;
			if (PyArray_TYPE(item)==NPY_INT)
				return STRING_INT;
			if (PyArray_TYPE(item)==NPY_DOUBLE)
				return STRING_SHORT;
			if (PyArray_TYPE(item)==NPY_USHORT)
				return STRING_WORD;
		}
	}

	return UNDEFINED;
}


INT CPythonInterface::get_int()
{
	const PyObject* i=get_arg_increment();
	if (!i || !PyInt_Check(i))
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return PyInt_AS_LONG(i);
}

DREAL CPythonInterface::get_real()
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

	return PyInt_AS_LONG(b) != 0;
}


CHAR* CPythonInterface::get_string(INT& len)
{
	const PyObject* s=get_arg_increment();
	if (!s || !PyString_Check(s))
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	len=PyString_Size((PyObject*) s);
	CHAR* str=PyString_AS_STRING(s);
	ASSERT(str && len>0);

	CHAR* cstr = new CHAR[len+1];
	ASSERT(cstr);

	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	return cstr;
}

#define GET_VECTOR(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(sg_type*& vector, INT& len)			\
{ 																			\
	const PyArrayObject* py_vec=(PyArrayObject *) get_arg_increment();		\
	if (!PyArray_Check(py_vec) || py_vec->nd!=1 || PyArray_TYPE(py_vec)!=py_type)	\
		SG_ERROR("Expected " error_string " Vector as argument %d\n",		\
			m_rhs_counter); 												\
																			\
	len=py_vec->dimensions[0]; 												\
	vector=new sg_type[len];												\
	ASSERT(vector);															\
	if_type* data=(if_type*) py_vec->data;									\
																			\
	for (INT i=0; i<len; i++)												\
			vector[i]=data[i];												\
}

GET_VECTOR(get_byte_vector, NPY_BYTE, BYTE, BYTE, "Byte")
GET_VECTOR(get_char_vector, NPY_CHAR, CHAR, char, "Char")
GET_VECTOR(get_int_vector, NPY_INT, INT, int, "Integer")
GET_VECTOR(get_short_vector, NPY_SHORT, SHORT, short, "Short")
GET_VECTOR(get_shortreal_vector, NPY_FLOAT, SHORTREAL, float, "Single Precision")
GET_VECTOR(get_real_vector, NPY_DOUBLE, DREAL, double, "Double Precision")
GET_VECTOR(get_word_vector, NPY_USHORT, WORD, unsigned short, "Word")
#undef GET_VECTOR


#define GET_MATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(sg_type*& matrix, INT& num_feat, INT& num_vec)	\
{ 																			\
	const PyArrayObject* py_mat=(PyArrayObject *) get_arg_increment(); 		\
	if (!PyArray_Check(py_mat) || PyArray_TYPE(py_mat)!=py_type) 			\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n",		\
			m_rhs_counter); 												\
 																			\
	num_vec=py_mat->dimensions[0]; 											\
	num_feat=py_mat->nd; 													\
	matrix=new sg_type[num_vec*num_feat]; 									\
	ASSERT(matrix); 														\
	if_type* data=(if_type*) py_mat->data; 									\
 																			\
	for (INT i=0; i<num_vec; i++) 											\
		for (INT j=0; j<num_feat; j++) 										\
			matrix[i*num_feat+j]=data[i*num_feat+j];						\
}

GET_MATRIX(get_byte_matrix, NPY_BYTE, BYTE, BYTE, "Byte")
GET_MATRIX(get_char_matrix, NPY_CHAR, CHAR, char, "Char")
GET_MATRIX(get_int_matrix, NPY_INT, INT, int, "Integer")
GET_MATRIX(get_short_matrix, NPY_SHORT, SHORT, short, "Short")
GET_MATRIX(get_shortreal_matrix, NPY_FLOAT, SHORTREAL, float, "Single Precision")
GET_MATRIX(get_real_matrix, NPY_DOUBLE, DREAL, double, "Double Precision")
GET_MATRIX(get_word_matrix, NPY_USHORT, WORD, unsigned short, "Word")
#undef GET_MATRIX


#define GET_SPARSEMATRIX(function_name, py_type, sg_type, if_type, error_string) \
void CPythonInterface::function_name(TSparse<sg_type>*& matrix, INT& num_feat, INT& num_vec) \
{																			\
	/* no sparse available yet */ \
	return; \
	\
	/* \
	const PyArray_Object* py_mat=(PyArrayObject *) get_arg_increment(); 	\
	if (!PyArray_Check(py_mat)) 											\
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter); \
 																			\
	if (!PyArray_TYPE(py_mat)!=py_type) 									\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n",		\
			m_rhs_counter); 												\
 																			\
	num_vec=py_mat->dimensions[0]; 											\
	num_feat=py_mat->nd; 													\
	matrix=new TSparse<sg_type>[num_vec]; 									\
	ASSERT(matrix); 														\
	if_type* data=(if_type*) py_mat->data; 									\
 																			\
	LONG nzmax=mxGetNzmax(mx_mat); 											\
	mwIndex* ir=mxGetIr(mx_mat); 											\
	mwIndex* jc=mxGetJc(mx_mat); 											\
	LONG offset=0; 															\
	for (INT i=0; i<num_vec; i++) 											\
	{ 																		\
		INT len=jc[i+1]-jc[i]; 												\
		matrix[i].vec_index=i; 												\
		matrix[i].num_feat_entries=len;										\
 																			\
		if (len>0) 															\
		{ 																	\
			matrix[i].features=new TSparseEntry<sg_type>[len]; 				\
			ASSERT(matrix[i].features); 									\
 																			\
			for (INT j=0; j<len; j++) 										\
			{ 																\
				matrix[i].features[j].entry=data[offset]; 					\
				matrix[i].features[j].feat_index=ir[offset]; 				\
				offset++; 													\
			} 																\
		} 																	\
		else 																\
			matrix[i].features=NULL; 										\
	} 																		\
	ASSERT(offset==nzmax); 													\
	*/ \
}

GET_SPARSEMATRIX(get_real_sparsematrix, NPY_DOUBLE, DREAL, double, "Double Precision")
/*  future versions might support types other than DREAL
GET_SPARSEMATRIX(get_byte_sparsematrix, "uint8", BYTE, BYTE, "Byte")
GET_SPARSEMATRIX(get_char_sparsematrix, "char", CHAR, mxChar, "Char")
GET_SPARSEMATRIX(get_int_sparsematrix, "int32", INT, int, "Integer")
GET_SPARSEMATRIX(get_short_sparsematrix, "int16", SHORT, short, "Short")
GET_SPARSEMATRIX(get_shortreal_sparsematrix, "single", SHORTREAL, float, "Single Precision")
GET_SPARSEMATRIX(get_word_sparsematrix, "uint16", WORD, unsigned short, "Word")*/
#undef GET_SPARSEMATRIX


#define GET_STRINGLIST(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(T_STRING<sg_type>*& strings, INT& num_str, INT& max_string_len)	\
{ 																			\
	const PyArrayObject* py_str=(PyArrayObject *) get_arg_increment();		\
	if (!PyArray_Check(py_str))												\
		SG_ERROR("Expected Stringlist as argument (none given).\n");		\
																			\
	if (PyList_Check(py_str))												\
	{																		\
		num_str=py_str->nd;													\
		ASSERT(num_str>=1);													\
																			\
		strings=new T_STRING<sg_type>[num_str];								\
		ASSERT(strings);													\
																			\
		for (int i=0; i<num_str; i++)										\
		{																	\
			const PyArrayObject* str=										\
				(PyArrayObject *) PyList_GET_ITEM(py_str, i);				\
			if (!PyArray_Check(str) || PyArray_TYPE(str)!=py_type ||		\
				str->nd==1)													\
				SG_ERROR("Expected String of type " error_string " as argument %d.\n",	\
					m_rhs_counter);											\
																			\
			INT len=str->dimensions[0];										\
			if (len>0) 														\
			{ 																\
				if_type* data=(if_type*) str->data;							\
				strings[i].length=len; /* all must have same length */ 		\
				strings[i].string=new sg_type[len+1]; /* not zero terminated */	\
				ASSERT(strings[i].string); 									\
				INT j; 														\
				for (j=0; j<len; j++) 										\
					strings[i].string[j]=data[j]; 							\
				strings[i].string[j]='\0'; 									\
				max_string_len=CMath::max(max_string_len, len);				\
			}																\
			else															\
			{																\
				SG_WARNING( "string with index %d has zero length.\n", i+1);	\
				strings[i].length=0;										\
				strings[i].string=NULL;										\
			}																\
		}																	\
	}																		\
	else if (PyArray_TYPE(py_str)==py_type)									\
	{																		\
		if_type* data=(if_type*) py_str->data;								\
		num_str=py_str->nd; 												\
		INT len=py_str->dimensions[0]; 										\
		strings=new T_STRING<sg_type>[num_str]; 							\
		ASSERT(strings); 													\
																			\
		for (INT i=0; i<num_str; i++) 										\
		{ 																	\
			if (len>0) 														\
			{ 																\
				strings[i].length=len; /* all must have same length*/		\
				strings[i].string=new sg_type[len+1]; /* not zero terminated */	\
				ASSERT(strings[i].string); 									\
				INT j; 														\
				for (j=0; j<len; j++) 										\
					strings[i].string[j]=data[j+i*len]; 					\
				strings[i].string[j]='\0'; 									\
			} 																\
			else 															\
			{ 																\
				SG_WARNING( "string with index %d has zero length.\n", i+1);	\
				strings[i].length=0; 										\
				strings[i].string=NULL; 									\
			} 																\
		} 																	\
		max_string_len=len;													\
	}																		\
	else																	\
		SG_ERROR("Expected String as argument %d.\n", m_rhs_counter);		\
}

GET_STRINGLIST(get_byte_string_list, NPY_BYTE, BYTE, BYTE, "Byte")
GET_STRINGLIST(get_char_string_list, NPY_CHAR, CHAR, char, "Char")
GET_STRINGLIST(get_int_string_list, NPY_INT, INT, int, "Integer")
GET_STRINGLIST(get_short_string_list, NPY_SHORT, SHORT, short, "Short")
GET_STRINGLIST(get_word_string_list, NPY_USHORT, WORD, unsigned short, "Word")
#undef GET_STRINGLIST



/** set functions - to pass data from shogun to the target interface */
void CPythonInterface::create_return_values(INT num_val)
{
}

#define SET_VECTOR(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const sg_type* vector, INT len)		\
{																			\
	if (!vector)															\
		SG_ERROR("Given vector is invalid.\n");								\
																			\
	npy_intp* dims=new npy_intp[len];										\
	ASSERT(dims);															\
	PyObject* py_vec=PyArray_SimpleNew(1, dims, py_type);					\
	if (!PyArray_Check(py_vec))												\
		SG_ERROR("Couldn't create " error_string " Vector of length %d.\n",	\
			len);															\
																			\
	if_type* data=(if_type*) ((PyArrayObject *) py_vec)->data;				\
																			\
	for (INT i=0; i<len; i++)												\
		data[i]=vector[i];													\
																			\
	set_arg_increment(py_vec);												\
}

SET_VECTOR(set_byte_vector, NPY_BYTE, BYTE, BYTE, "Byte")
SET_VECTOR(set_char_vector, NPY_CHAR, CHAR, char, "Char")
SET_VECTOR(set_int_vector, NPY_INT, INT, int, "Integer")
SET_VECTOR(set_short_vector, NPY_SHORT, SHORT, short, "Short")
SET_VECTOR(set_shortreal_vector, NPY_FLOAT, SHORTREAL, float, "Single Precision")
SET_VECTOR(set_real_vector, NPY_DOUBLE, DREAL, double, "Double Precision")
SET_VECTOR(set_word_vector, NPY_USHORT, WORD, unsigned short, "Word")
#undef SET_VECTOR


#define SET_MATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const sg_type* matrix, INT num_feat, INT num_vec)	\
{ 																			\
	if (!matrix) 															\
		SG_ERROR("Given matrix is invalid.\n");								\
 																			\
	npy_intp* dims=new npy_intp[num_vec];									\
	ASSERT(dims);															\
	PyObject* py_mat=PyArray_SimpleNew(num_feat, dims, py_type);			\
	if (!PyArray_Check(py_mat)) 											\
		SG_ERROR("Couldn't create " error_string " Matrix of %d rows and %d cols.\n",	\
			num_feat, num_vec);												\
 																			\
	if_type* data=(if_type*) ((PyArrayObject *) py_mat)->data; 				\
 																			\
	for (INT i=0; i<num_vec; i++) 											\
		for (INT j=0; j<num_feat; j++) 										\
			data[i*num_feat+j]=matrix[i*num_feat+j]; 						\
 																			\
	set_arg_increment(py_mat); 												\
}

SET_MATRIX(set_byte_matrix, NPY_BYTE, BYTE, BYTE, "Byte")
SET_MATRIX(set_char_matrix, NPY_CHAR, CHAR, char, "Char")
SET_MATRIX(set_int_matrix, NPY_INT, INT, int, "Integer")
SET_MATRIX(set_short_matrix, NPY_SHORT, SHORT, short, "Short")
SET_MATRIX(set_shortreal_matrix, NPY_FLOAT, SHORTREAL, float, "Single Precision")
SET_MATRIX(set_real_matrix, NPY_DOUBLE, DREAL, double, "Double Precision")
SET_MATRIX(set_word_matrix, NPY_USHORT, WORD, unsigned short, "Word")
#undef SET_MATRIX

#define SET_SPARSEMATRIX(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const TSparse<sg_type>* matrix, INT num_feat, INT num_vec, LONG nnz)	\
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
	LONG offset=0;															\
	for (INT i=0; i<num_vec; i++)											\
	{																		\
		INT len=matrix[i].num_feat_entries;									\
		jc[i]=offset;														\
		for (INT j=0; j<len; j++)											\
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

SET_SPARSEMATRIX(set_real_sparsematrix, NPY_DOUBLE, DREAL, double, "Double Precision")

/* future version might support this
SET_SPARSEMATRIX(set_byte_sparsematrix, mxUINT8_CLASS, BYTE, BYTE, "Byte")
SET_SPARSEMATRIX(set_char_sparsematrix, mxCHAR_CLASS, CHAR, mxChar, "Char")
SET_SPARSEMATRIX(set_int_sparsematrix, mxINT32_CLASS, INT, int, "Integer")
SET_SPARSEMATRIX(set_short_sparsematrix, mxINT16_CLASS, SHORT, short, "Short")
SET_SPARSEMATRIX(set_shortreal_sparsematrix, mxSINGLE_CLASS, SHORTREAL, float, "Single Precision")
SET_SPARSEMATRIX(set_word_sparsematrix, mxUINT16_CLASS, WORD, unsigned short, "Word")*/
#undef SET_SPARSEMATRIX


#define SET_STRINGLIST(function_name, py_type, sg_type, if_type, error_string)	\
void CPythonInterface::function_name(const T_STRING<sg_type>* strings, INT num_str)	\
{																				\
	if (!strings)																\
		SG_ERROR("Given strings are invalid.\n");								\
																				\
	PyObject* py_str=PyList_New(num_str);										\
	if (!PyArray_Check(py_str)) 												\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str);		\
																				\
	for (INT i=0; i<num_str; i++)												\
	{																			\
		INT len=strings[i].length;												\
		if (len>0)																\
		{																		\
			npy_intp* dims=new npy_intp[num_str];								\
			ASSERT(dims);														\
			PyObject* str=PyArray_SimpleNew(1, dims, py_type);					\
			if (!PyArray_Check(str)) 											\
				SG_ERROR("Couldn't create " error_string " String %d of length %d.\n",	\
					i, len);													\
																				\
			if_type* data=(if_type*) ((PyArrayObject *) str)->data;				\
																				\
			for (INT j=0; j<len; j++)											\
				data[j]=strings[i].string[j];									\
			PyList_SET_ITEM(py_str, i, str);									\
		}																		\
	}																			\
																				\
	set_arg_increment(py_str);													\
}

SET_STRINGLIST(set_byte_string_list, NPY_BYTE, BYTE, BYTE, "Byte")
SET_STRINGLIST(set_char_string_list, NPY_CHAR, CHAR, char, "Char")
SET_STRINGLIST(set_int_string_list, NPY_INT, INT, int, "Integer")
SET_STRINGLIST(set_short_string_list, NPY_SHORT, SHORT, short, "Short")
SET_STRINGLIST(set_word_string_list, NPY_USHORT, WORD, unsigned short, "Word")
#undef SET_STRINGLIST


PyObject* sg(PyObject* self, PyObject* args)
{
	delete interface;
	interface=new CPythonInterface(self, args);

	try
	{
		if (!interface->handle())
			SG_ERROR("interface currently does not handle this command.\n");
	}
	catch (ShogunException e)
	{
		Py_INCREF(Py_None);
		return Py_None;
	}

	return ((CPythonInterface*) interface)->get_return_values();
}

void exitsg(void)
{
	SG_SINFO( "quitting...\n");
	//delete gui;
}

static PyMethodDef sg_pythonmethods[] = {
    {(char*) "sg",  sg, METH_VARARGS, (char*) "Shogun."},
    {NULL, NULL, 0, NULL}        /* Sentinel */
};

PyMODINIT_FUNC initsg(void)
{
	// initialize python interpreter
	Py_Initialize();

	// initialize threading (just in case it is needed)
	PyEval_InitThreads();

	// initialize textgui
	//gui=new CTextGUI(0, NULL) ;

    // callback to cleanup at exit
	Py_AtExit(exitsg);

	// initialize callbacks
    Py_InitModule((char*) "sg", sg_pythonmethods);
}

#endif // HAVE_PYTHON && !HAVE_SWIG
