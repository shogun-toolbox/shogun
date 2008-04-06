#include "lib/config.h"

#if defined(HAVE_PYTHON) && !defined(HAVE_SWIG)

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/python.h"

#include "interface/SGInterface.h"
#include "interface/PythonInterface.h"

extern "C" {
#include <numpy/arrayobject.h>
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
}

CPythonInterface::~CPythonInterface()
{
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

		if (PyString_Check(item))
			return STRING_CHAR;
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
	if (!py_vec || !PyArray_Check(py_vec) || py_vec->nd!=1 ||				\
			PyArray_TYPE(py_vec)!=py_type)									\
	{																		\
		SG_ERROR("Expected " error_string " Vector as argument %d\n",		\
			m_rhs_counter); 												\
	}																		\
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
	if (!py_mat || !PyArray_Check(py_mat) || 								\
			PyArray_TYPE(py_mat)!=py_type || py_mat->nd!=2) 				\
	{																		\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n",		\
			m_rhs_counter); 												\
	}																		\
 																			\
	num_feat=py_mat->dimensions[0]; 										\
	num_vec=py_mat->dimensions[1]; 											\
	matrix=new sg_type[num_vec*num_feat]; 									\
	ASSERT(matrix); 														\
																			\
	char* data=py_mat->data; 												\
	npy_intp* strides= py_mat->strides; 									\
	npy_intp d2_offs=0;														\
	for (INT i=0; i<num_feat; i++) 											\
	{																		\
		npy_intp offs=d2_offs;												\
		for (INT j=0; j<num_vec; j++) 										\
		{																	\
			matrix[i+j*num_feat]=*((if_type*)(data+offs));					\
			offs+=strides[1];												\
		}																	\
		d2_offs+=strides[0];												\
	}																		\
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


#define GET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(T_STRING<sg_type>*& strings, INT& num_str, INT& max_string_len)	\
{ 																			\
	max_string_len=0;														\
	const PyObject* py_str= get_arg_increment();									\
	if (!py_str)															\
		SG_ERROR("Expected Stringlist as argument (none given).\n");		\
																			\
	if (PyList_Check(py_str))												\
	{																		\
		if (!is_char_str)													\
			SG_ERROR("Only Character Strings supported.\n");				\
																			\
        num_str=PyList_Size((PyObject*) py_str);										\
		ASSERT(num_str>=1);													\
																			\
		strings=new T_STRING<sg_type>[num_str];								\
		ASSERT(strings);													\
																			\
		for (int i=0; i<num_str; i++)										\
		{																	\
            PyObject *o = PyList_GetItem((PyObject*) py_str,i);				\
            if (PyString_Check(o))											\
            {																\
                INT len=PyString_Size(o);									\
                const sg_type* str= (const sg_type*) PyString_AsString(o);	\
																			\
                strings[i].length=len;										\
                strings[i].string=NULL;										\
				max_string_len=CMath::max(max_string_len, len);				\
																			\
                if (len>0)													\
                {															\
                    strings[i].string=new sg_type[len+1];					\
                    memcpy(strings[i].string, str, len);					\
					strings[i].string[len]='\0';							\
                }															\
            }																\
            else															\
            {																\
                for (INT j=0; j<i; j++)										\
                    delete[] strings[i].string;								\
                delete[] strings;											\
                SG_ERROR("All elements in list must be strings, error in line %d.\n", i);\
            }																\
		}																	\
	}																		\
	else if (PyArray_TYPE(py_str)==py_type && ((PyArrayObject*) py_str)->nd==2)	\
	{																		\
		const PyArrayObject* py_array_str=(const PyArrayObject*) py_str;	\
		if_type* data=(if_type*) py_array_str->data;						\
		num_str=py_array_str->dimensions[0]; 								\
		INT len=py_array_str->dimensions[1]; 								\
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

GET_STRINGLIST(get_byte_string_list, NPY_BYTE, BYTE, BYTE, 1, "Byte")
GET_STRINGLIST(get_char_string_list, NPY_CHAR, CHAR, char, 1, "Char")
GET_STRINGLIST(get_int_string_list, NPY_INT, INT, int, 0, "Integer")
GET_STRINGLIST(get_short_string_list, NPY_SHORT, SHORT, short, 0, "Short")
GET_STRINGLIST(get_word_string_list, NPY_USHORT, WORD, unsigned short, 0, "Word")
#undef GET_STRINGLIST



/** set functions - to pass data from shogun to the target interface */

void CPythonInterface::set_int(INT scalar)
{
	PyObject* o=Py_BuildValue("i", scalar);
	if (!o)
		SG_ERROR("Could not build an integer.\n");

	set_arg_increment(o);
}

void CPythonInterface::set_real(DREAL scalar)
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
void CPythonInterface::function_name(const sg_type* vector, INT len)		\
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
	if (!matrix || num_feat<1 || num_vec<1) 								\
		SG_ERROR("Given matrix is invalid.\n");								\
 																			\
	npy_intp dims[2]={num_feat, num_vec};									\
	PyObject* py_mat=PyArray_SimpleNew(2, dims, py_type);					\
	if (!py_mat || !PyArray_Check(py_mat)) 									\
		SG_ERROR("Couldn't create " error_string " Matrix of %d rows and %d cols.\n",	\
			num_feat, num_vec);												\
	ASSERT(PyArray_ISCARRAY(py_mat));										\
 																			\
	if_type* data=(if_type*) ((PyArrayObject *) py_mat)->data; 				\
 																			\
	for (INT j=0; j<num_feat; j++) 											\
		for (INT i=0; i<num_vec; i++) 										\
			data[i+j*num_vec]=matrix[i*num_feat+j]; 						\
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


#define SET_STRINGLIST(function_name, py_type, sg_type, if_type, is_char_str, error_string)	\
void CPythonInterface::function_name(const T_STRING<sg_type>* strings, INT num_str)	\
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
	for (INT i=0; i<num_str; i++)												\
	{																			\
		INT len=strings[i].length;												\
		if (len>0)																\
		{																		\
			PyObject* str=PyString_FromStringAndSize((const char*) strings[i].string, len); \
			if (!str) 															\
				SG_ERROR("Couldn't create " error_string 						\
						" String %d of length %d.\n", i, len);					\
																				\
			PyList_SET_ITEM(py_str, i, str);									\
		}																		\
	}																			\
																				\
	set_arg_increment(py_str);													\
}

SET_STRINGLIST(set_byte_string_list, NPY_BYTE, BYTE, BYTE, 0, "Byte")
SET_STRINGLIST(set_char_string_list, NPY_CHAR, CHAR, char, 1, "Char")
SET_STRINGLIST(set_int_string_list, NPY_INT, INT, int, 0, "Integer")
SET_STRINGLIST(set_short_string_list, NPY_SHORT, SHORT, short, 0, "Short")
SET_STRINGLIST(set_word_string_list, NPY_USHORT, WORD, unsigned short, 0, "Word")
#undef SET_STRINGLIST


bool CPythonInterface::create_return_values(INT num)
{
	if (num<=0)
		return true;

	m_lhs=PyTuple_New(num);
	ASSERT(m_lhs);

	m_nlhs=num;
	return PyTuple_GET_SIZE(m_lhs)==num;
}


PyObject* sg(PyObject* self, PyObject* args)
{
	delete interface;
	interface=new CPythonInterface(self, args);

	try
	{
		if (!interface->handle())
			SG_ERROR("Unknown command.\n");
	}
	catch (ShogunException e)
	{
		return NULL;
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
	import_array();
}

#endif // HAVE_PYTHON && !HAVE_SWIG
