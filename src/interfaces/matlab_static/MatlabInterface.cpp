// somehow this has to depend on the matlab version
// comment it out for now
//
// #include <mexversion.c>
//

#include "MatlabInterface.h"

#include <stdio.h>
#include <ui/SGInterface.h>
#include <lib/ShogunException.h>
#include <lib/config.h>
#include <base/init.h>

#ifdef HAVE_PYTHON
#include "../python/PythonInterface.h"
#endif

#ifdef HAVE_OCTAVE
#include "../octave/OctaveInterface.h"
#endif

#ifdef HAVE_R
#include "../r/RInterface.h"
#endif

void matlab_print_message(FILE* target, const char* str)
{
	if (target==stdout)
		mexPrintf("%s", str);
	else
		fprintf(target, "%s", str);
}

void matlab_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		mexWarnMsgTxt(str);
	else
		fprintf(target, "%s", str);
}

void matlab_print_error(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void matlab_cancel_computations(bool &delayed, bool &immediately)
{
}

extern CSGInterface* interface;

CMatlabInterface::CMatlabInterface(
	int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
: CSGInterface()
{
	reset(nlhs, plhs, nrhs, prhs);

}

CMatlabInterface::~CMatlabInterface()
{
}

void CMatlabInterface::reset(
	int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	CSGInterface::reset();

	m_nlhs=nlhs;
	m_nrhs=nrhs;
	m_lhs=plhs;
	m_rhs=prhs;
}


/** get functions - to pass data from the target interface to shogun */

/// get type of current argument (does not increment argument counter)
IFType CMatlabInterface::get_argument_type()
{
	const mxArray* arg=m_rhs[m_rhs_counter];
	ASSERT(arg);

	if (mxIsSparse(arg))
	{
		if (mxIsUint8(arg))
			return SPARSE_BYTE;
		if (mxIsChar(arg))
			return SPARSE_CHAR;
		if (mxIsInt32(arg))
			return SPARSE_INT;
		if (mxIsDouble(arg))
			return SPARSE_REAL;
		if (mxIsInt16(arg))
			return SPARSE_SHORT;
		if (mxIsSingle(arg))
			return SPARSE_SHORTREAL;
		if (mxIsUint16(arg))
			return SPARSE_WORD;

		return UNDEFINED;
	}

	if (mxIsInt32(arg))
		return DENSE_INT;
	if (mxIsDouble(arg))
		return DENSE_REAL;
	if (mxIsInt16(arg))
		return DENSE_SHORT;
	if (mxIsSingle(arg))
		return DENSE_SHORTREAL;
	if (mxIsUint16(arg))
		return DENSE_WORD;

	if (mxIsChar(arg))
		return STRING_CHAR;
	if (mxIsUint8(arg))
		return STRING_BYTE;

	if (mxIsCell(arg))
	{
		const mxArray* cell=mxGetCell(arg, 0);
		if (cell && mxGetM(cell)==1)
		{
			if (mxIsUint8(cell))
				return STRING_BYTE;
			if (mxIsChar(cell))
				return STRING_CHAR;
			if (mxIsInt32(cell))
				return STRING_INT;
			if (mxIsInt16(cell))
				return STRING_SHORT;
			if (mxIsUint16(cell))
				return STRING_WORD;
		}
	}

	return UNDEFINED;
}


int32_t CMatlabInterface::get_int()
{
	const mxArray* i=get_arg_increment();
	if (!i || !mxIsNumeric(i) || mxGetN(i)!=1 || mxGetM(i)!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	double s=mxGetScalar(i);
	if (s-CMath::floor(s)!=0)
		SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);

	return int32_t(s);
}

float64_t CMatlabInterface::get_real()
{
	const mxArray* f=get_arg_increment();
	if (!f || !mxIsNumeric(f) || mxGetN(f)!=1 || mxGetM(f)!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return mxGetScalar(f);
}

bool CMatlabInterface::get_bool()
{
	const mxArray* b=get_arg_increment();
	if (mxIsLogical(b) && mxGetN(b)==1 && mxGetM(b)==1)
		return mxGetLogicals(b)[0];
	else if (mxIsNumeric(b))
		return (mxGetScalar(b)!=0);
	else
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return false;
}


char* CMatlabInterface::get_string(int32_t& len)
{
	bool zero_terminate=true;
	const mxArray* s=get_arg_increment();

	if ( !(mxIsChar(s)) || (mxGetM(s)!=1) )
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	len=mxGetN(s);
	char* string=NULL;
	if (zero_terminate)
		string=SG_MALLOC(char, len+1);
	else
		string=SG_MALLOC(char, len);
	mxChar* c=mxGetChars(s);
	ASSERT(c);
	for (int32_t i=0; i<len; i++)
		string[i]= (char) (c[i]);

	if (zero_terminate)
		string[len]='\0';

	return string;
}

#define GET_VECTOR(function_name, mx_type, sg_type, if_type, error_string)	\
void CMatlabInterface::function_name(sg_type*& vector, int32_t& len)	\
{																	\
	const mxArray* mx_vec=get_arg_increment();						\
	if (!mx_vec || mxGetM(mx_vec)!=1 || !mxIsClass(mx_vec, mx_type))		\
		SG_ERROR("Expected " error_string " (1xN) Vector, got vector of " \
				"class %s shape (%dx%d) as argument %d\n", \
			mxGetClassName(mx_vec), mxGetM(mx_vec), mxGetN(mx_vec), m_rhs_counter);				\
																	\
	len=mxGetNumberOfElements(mx_vec);								\
	vector=SG_MALLOC(sg_type, len);										\
	if_type* data=(if_type*) mxGetData(mx_vec);						\
																	\
	for (int32_t i=0; i<len; i++)										\
			vector[i]=data[i];										\
}

GET_VECTOR(get_vector, "uint8", uint8_t, uint8_t, "Byte")
GET_VECTOR(get_vector, "char", char, mxChar, "Char")
GET_VECTOR(get_vector, "int32", int32_t, int, "Integer")
GET_VECTOR(get_vector, "int16", int16_t, short, "Short")
GET_VECTOR(get_vector, "single", float32_t, float, "Single Precision")
GET_VECTOR(get_vector, "double", float64_t, double, "Double Precision")
GET_VECTOR(get_vector, "uint16", uint16_t, unsigned short, "Word")
#undef GET_VECTOR

#define GET_MATRIX(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{																				\
	const mxArray* mx_mat=get_arg_increment();									\
	if (!mx_mat || !(mxIsClass(mx_mat, mx_type)))								\
		SG_ERROR("Expected " error_string " Matrix, got class %s as argument %d\n", \
			mxGetClassName(mx_mat), m_rhs_counter);							\
																				\
	num_vec=mxGetN(mx_mat);													\
	num_feat=mxGetM(mx_mat);													\
	matrix=SG_MALLOC(sg_type, num_vec*num_feat);										\
	if_type* data=(if_type*) mxGetData(mx_mat);								\
																				\
	for (int32_t i=0; i<num_vec; i++)												\
		for (int32_t j=0; j<num_feat; j++)											\
			matrix[i*num_feat+j]=data[i*num_feat+j];							\
}

GET_MATRIX(get_matrix, "uint8", uint8_t, uint8_t, "Byte")
GET_MATRIX(get_matrix, "char", char, mxChar, "Char")
GET_MATRIX(get_matrix, "int32", int32_t, int, "Integer")
GET_MATRIX(get_matrix, "int16", int16_t, short, "Short")
GET_MATRIX(get_matrix, "single", float32_t, float, "Single Precision")
GET_MATRIX(get_matrix, "double", float64_t, double, "Double Precision")
GET_MATRIX(get_matrix, "uint16", uint16_t, unsigned short, "Word")
#undef GET_MATRIX

#define GET_NDARRAY(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(sg_type*& array, int32_t*& dims, int32_t& num_dims)\
{																				\
	const mxArray* mx_array=get_arg_increment();								\
	if (!mx_array || !(mxIsClass(mx_array, mx_type)))							\
		SG_ERROR("Expected " error_string " ND Array, got class %s as argument %d\n", \
			mxGetClassName(mx_array), m_rhs_counter);							\
																				\
	num_dims = (int32_t) mxGetNumberOfDimensions(mx_array);							\
	const mwSize* mw_dims = mxGetDimensions(mx_array);							\
	mwSize total_size=mxGetNumberOfElements(mx_array);							\
	if_type* data=(if_type*) mxGetData(mx_array);								\
																				\
	dims=SG_MALLOC(int32_t, num_dims);														\
	for (int32_t d=0; d<num_dims; d++)												\
		dims[d]=(int32_t) mw_dims[d];												\
																				\
	array=SG_MALLOC(sg_type, total_size);												\
	for (mwSize i=0; i<total_size; i++)										\
		array[i]=data[i];														\
}

GET_NDARRAY(get_ndarray, "uint8", uint8_t, uint8_t, "Byte")
GET_NDARRAY(get_ndarray, "char", char, mxChar, "Char")
GET_NDARRAY(get_ndarray, "int32", int32_t, int, "Integer")
GET_NDARRAY(get_ndarray, "int16", int16_t, short, "Short")
GET_NDARRAY(get_ndarray, "single", float32_t, float, "Single Precision")
GET_NDARRAY(get_ndarray, "double", float64_t, double, "Double Precision")
GET_NDARRAY(get_ndarray, "uint16", uint16_t, unsigned short, "Word")
#undef GET_NDARRAY

#define GET_SPARSEMATRIX(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec) \
{																						\
	const mxArray* mx_mat=get_arg_increment();											\
	if (!mx_mat || !mxIsSparse(mx_mat))												\
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);			\
																						\
	if (!mxIsClass(mx_mat,mx_type))													\
		SG_ERROR("Expected " error_string " Matrix, got class %s as argument %d\n",	\
			mxGetClassName(mx_mat), m_rhs_counter);									\
																						\
	num_vec=mxGetN(mx_mat);															\
	num_feat=mxGetM(mx_mat);															\
	matrix=SG_MALLOC(SGSparseVector<sg_type>, num_vec);												\
	if_type* data=(if_type*) mxGetData(mx_mat);										\
																						\
	int64_t nzmax=mxGetNzmax(mx_mat);														\
	mwIndex* ir=mxGetIr(mx_mat);														\
	mwIndex* jc=mxGetJc(mx_mat);														\
	int64_t offset=0;																		\
	for (int32_t i=0; i<num_vec; i++)														\
	{																					\
		int32_t len=jc[i+1]-jc[i];															\
		matrix[i].num_feat_entries=len;												\
																						\
		if (len>0)																		\
		{																				\
			matrix[i].features=SG_MALLOC(SGSparseVectorEntry<sg_type>, len);							\
																						\
			for (int32_t j=0; j<len; j++)													\
			{																			\
				matrix[i].features[j].entry=data[offset];								\
				matrix[i].features[j].feat_index=ir[offset];							\
				offset++;																\
			}																			\
		}																				\
		else																			\
			matrix[i].features=NULL;													\
	}																					\
	ASSERT(offset==nzmax);																\
}

GET_SPARSEMATRIX(get_sparse_matrix, "double", float64_t, double, "Double Precision")
/*  future versions might support types other than float64_t
GET_SPARSEMATRIX(get_sparse_matrix, "uint8", uint8_t, uint8_t, "Byte")
GET_SPARSEMATRIX(get_sparse_matrix, "char", char, mxChar, "Char")
GET_SPARSEMATRIX(get_sparse_matrix, "int32", int32_t, int, "Integer")
GET_SPARSEMATRIX(get_sparse_matrix, "int16", int16_t, short, "Short")
GET_SPARSEMATRIX(get_sparse_matrix, "single", float32_t, float, "Single Precision")
GET_SPARSEMATRIX(get_sparse_matrix, "uint16", uint16_t, unsigned short, "Word")*/
#undef GET_SPARSEMATRIX


#define GET_STRINGLIST(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len)	\
{																						\
	const mxArray* mx_str=get_arg_increment();											\
	if (!mx_str)																		\
		SG_ERROR("Expected Stringlist as argument (none given).\n");					\
																						\
	if (mxIsCell(mx_str))																\
	{																					\
		num_str=mxGetNumberOfElements(mx_str);											\
		ASSERT(num_str>=1);																\
																						\
		strings=SG_MALLOC(SGString<sg_type>, num_str);											\
		for (int32_t i=0; i<num_str; i++)													\
		{																				\
			mxArray* str=mxGetCell(mx_str, i);											\
			if (!str)					\
				SG_ERROR("argument empty");		\
			if (!str || !mxIsClass(str, mx_type) || !mxGetM(str)==1)					\
				SG_ERROR("Expected String of type " error_string " as argument %d.  M:%i N:%i\n", m_rhs_counter,mxGetM(str),mxGetN(str)); \
																						\
			int32_t len=mxGetN(str);														\
			if (len>0)																	\
			{																			\
				if_type* data=(if_type*) mxGetData(str);								\
				strings[i].slen=len; /* all must have same length in matlab */		\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated in matlab */ \
				int32_t j;																	\
				for (j=0; j<len; j++)													\
					strings[i].string[j]= (sg_type) data[j];							\
				strings[i].string[j]='\0';												\
				max_string_len=CMath::max(max_string_len, len);							\
			}																			\
			else																		\
			{																			\
				SG_WARNING( "string with index %d has zero length.\n", i+1);			\
				strings[i].slen=0;													\
				strings[i].string=NULL;													\
			}																			\
		}																				\
	}																					\
	else if (mxIsClass(mx_str, mx_type))												\
	{																					\
		if_type* data=(if_type*) mxGetData(mx_str);										\
		num_str=mxGetN(mx_str);														\
		int32_t len=mxGetM(mx_str);														\
		strings=SG_MALLOC(SGString<sg_type>, num_str);										\
																						\
		for (int32_t i=0; i<num_str; i++)													\
		{																				\
			if (len>0)																	\
			{																			\
				strings[i].slen=len; /* all must have same length in matlab */		\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated in matlab */ \
				int32_t j;																	\
				for (j=0; j<len; j++)													\
					strings[i].string[j]=(sg_type) data[j+i*len];						\
				strings[i].string[j]='\0';												\
			}																			\
			else																		\
			{																			\
				SG_WARNING( "string with index %d has zero length.\n", i+1);			\
				strings[i].slen=0;													\
				strings[i].string=NULL;												\
			}																			\
		}																				\
		max_string_len=len;																\
	}																					\
	else																				\
		SG_ERROR("Expected String, got class %s as argument %d.\n",						\
			mxGetClassName(mx_str), m_rhs_counter);										\
}

GET_STRINGLIST(get_string_list, "uint8", uint8_t, uint8_t, "Byte")
GET_STRINGLIST(get_string_list, "char", char, mxChar, "Char")
GET_STRINGLIST(get_string_list, "int32", int32_t, int, "Integer")
GET_STRINGLIST(get_string_list, "int16", int16_t, short, "Short")
GET_STRINGLIST(get_string_list, "uint16", uint16_t, unsigned short, "Word")
#undef GET_STRINGLIST


void CMatlabInterface::get_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* &attrs)
{
	attrs=NULL;
}

/** set functions - to pass data from shogun to the target interface */

void CMatlabInterface::set_int(int32_t scalar)
{
	mxArray* o=mxCreateNumericMatrix(1, 1, mxINT32_CLASS, mxREAL);
	if (!o)
		SG_ERROR("Couldn't create Integer.\n");

	int* data=(int*) mxGetData(o);
	data[0]=scalar;

	set_arg_increment(o);
}

void CMatlabInterface::set_real(float64_t scalar)
{
	mxArray* o=mxCreateNumericMatrix(1, 1, mxDOUBLE_CLASS, mxREAL);
	if (!o)
		SG_ERROR("Couldn't create Double.\n");

	double* data=(double*) mxGetData(o);
	data[0]=scalar;

	set_arg_increment(o);
}

void CMatlabInterface::set_bool(bool scalar)
{
	mxArray* o=mxCreateLogicalMatrix(1, 1);
	if (!o)
		SG_ERROR("Couldn't create Logical.\n");

	bool* data=(bool*) mxGetData(o);
	data[0]=scalar;

	set_arg_increment(o);
}


#define SET_VECTOR(function_name, mx_type, sg_type, if_type, error_string) \
void CMatlabInterface::function_name(const sg_type* vector, int32_t len)		\
{																			\
	if (!vector)															\
		SG_ERROR("Given vector is invalid.\n");								\
																			\
	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mx_type, mxREAL);			\
	if (!mx_vec)															\
		SG_ERROR("Couldn't create " error_string " Vector of length %d.\n", len);		\
																			\
	if_type* data=(if_type*) mxGetData(mx_vec);								\
																			\
	for (int32_t i=0; i<len; i++)												\
		data[i]=vector[i];													\
																			\
	set_arg_increment(mx_vec);												\
}

SET_VECTOR(set_vector, mxUINT8_CLASS, uint8_t, uint8_t, "Byte")
SET_VECTOR(set_vector, mxCHAR_CLASS, char, mxChar, "Char")
SET_VECTOR(set_vector, mxINT32_CLASS, int32_t, int, "Integer")
SET_VECTOR(set_vector, mxINT16_CLASS, int16_t, short, "Short")
SET_VECTOR(set_vector, mxSINGLE_CLASS, float32_t, float, "Single Precision")
SET_VECTOR(set_vector, mxDOUBLE_CLASS, float64_t, double, "Double Precision")
SET_VECTOR(set_vector, mxUINT16_CLASS, uint16_t, unsigned short, "Word")
#undef SET_VECTOR


#define SET_MATRIX(function_name, mx_type, sg_type, if_type, error_string)	\
void CMatlabInterface::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{																				\
	if (!matrix)																\
		SG_ERROR("Given matrix is invalid.\n");									\
																				\
	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mx_type, mxREAL);	\
	if (!mx_mat)																\
		SG_ERROR("Couldn't create " error_string " Matrix of %d rows and %d cols.\n", num_feat, num_vec); \
																				\
	if_type* data=(if_type*) mxGetData(mx_mat);								\
																				\
	for (int32_t i=0; i<num_vec; i++)												\
		for (int32_t j=0; j<num_feat; j++)											\
			data[i*num_feat+j]=matrix[i*num_feat+j];							\
																				\
	set_arg_increment(mx_mat);													\
}

SET_MATRIX(set_matrix, mxUINT8_CLASS, uint8_t, uint8_t, "Byte")
SET_MATRIX(set_matrix, mxCHAR_CLASS, char, mxChar, "Char")
SET_MATRIX(set_matrix, mxINT32_CLASS, int32_t, int, "Integer")
SET_MATRIX(set_matrix, mxINT16_CLASS, int16_t, short, "Short")
SET_MATRIX(set_matrix, mxSINGLE_CLASS, float32_t, float, "Single Precision")
SET_MATRIX(set_matrix, mxDOUBLE_CLASS, float64_t, double, "Double Precision")
SET_MATRIX(set_matrix, mxUINT16_CLASS, uint16_t, unsigned short, "Word")
#undef SET_MATRIX

#define SET_SPARSEMATRIX(function_name, mx_type, sg_type, if_type, error_string)	\
void CMatlabInterface::function_name(const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz) \
{																			\
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
}

SET_SPARSEMATRIX(set_sparse_matrix, mxDOUBLE_CLASS, float64_t, double, "Double Precision")

/* future version might support this
SET_SPARSEMATRIX(set_sparse_matrix, mxUINT8_CLASS, uint8_t, uint8_t, "Byte")
SET_SPARSEMATRIX(set_sparse_matrix, mxCHAR_CLASS, char, mxChar, "Char")
SET_SPARSEMATRIX(set_sparse_matrix, mxINT32_CLASS, int32_t, int, "Integer")
SET_SPARSEMATRIX(set_sparse_matrix, mxINT16_CLASS, int16_t, short, "Short")
SET_SPARSEMATRIX(set_sparse_matrix, mxSINGLE_CLASS, float32_t, float, "Single Precision")
SET_SPARSEMATRIX(set_sparse_matrix, mxUINT16_CLASS, uint16_t, unsigned short, "Word")*/
#undef SET_SPARSEMATRIX

#define SET_STRINGLIST(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(const SGString<sg_type>* strings, int32_t num_str)	\
{																					\
	if (!strings)																	\
		SG_ERROR("Given strings are invalid.\n");									\
																					\
	mxArray* mx_str= mxCreateCellMatrix(num_str, 1);								\
	if (!mx_str)																	\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str);			\
																					\
	for (int32_t i=0; i<num_str; i++)												\
	{																				\
		int32_t len=strings[i].slen;												\
		if (len>0)																	\
		{																			\
			mxArray* str=mxCreateNumericMatrix(1, len, mx_type, mxREAL);			\
			if (!str)																\
				SG_ERROR("Couldn't create " error_string " String %d of length %d.\n", i, len);		\
																					\
			if_type* data=(if_type*) mxGetData(str);								\
																					\
			for (int32_t j=0; j<len; j++)											\
				data[j]=strings[i].string[j];										\
			mxSetCell(mx_str, i, str);												\
		}																			\
	}																				\
																					\
	set_arg_increment(mx_str);														\
}

SET_STRINGLIST(set_string_list, mxUINT8_CLASS, uint8_t, uint8_t, "Byte")
SET_STRINGLIST(set_string_list, mxCHAR_CLASS, char, mxChar, "Char")
SET_STRINGLIST(set_string_list, mxINT32_CLASS, int32_t, int, "Integer")
SET_STRINGLIST(set_string_list, mxINT16_CLASS, int16_t, short, "Short")
SET_STRINGLIST(set_string_list, mxUINT16_CLASS, uint16_t, unsigned short, "Word")
#undef SET_STRINGLIST

void CMatlabInterface::set_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* attrs)
{
}

////////////////////////////////////////////////////////////////////

const mxArray* CMatlabInterface::get_arg_increment()
{
	const mxArray* retval;
	ASSERT(m_rhs_counter>=0 && m_rhs_counter<m_nrhs+1); // +1 for action
	ASSERT(m_rhs);

	retval=m_rhs[m_rhs_counter];
	m_rhs_counter++;

	return retval;
}

void CMatlabInterface::set_arg_increment(mxArray* mx_arg)
{
	ASSERT(m_lhs_counter>=0 && m_lhs_counter<m_nlhs);
	ASSERT(m_lhs);
	m_lhs[m_lhs_counter]=mx_arg;
	m_lhs_counter++;
}
////////////////////////////////////////////////////////////////////
//
bool CMatlabInterface::cmd_run_python()
{
#ifdef HAVE_PYTHON
	return CPythonInterface::run_python_helper(this);
#else
	return false;
#endif
}

bool CMatlabInterface::cmd_run_octave()
{
#ifdef HAVE_OCTAVE
	return COctaveInterface::run_octave_helper(this);
#else
	return false;
#endif
}

bool CMatlabInterface::cmd_run_r()
{
#ifdef HAVE_R
	return CRInterface::run_r_helper(this);
#else
	return false;
#endif
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
	try
	{
		if (!interface)
		{
			// init_shogun has to be called before anything else
			// exit_shogun is called upon destruction of the interface (see
			// destructor of CMatlabInterface
			init_shogun(&matlab_print_message, &matlab_print_warning,
					&matlab_print_error, &matlab_cancel_computations);
			interface=new CMatlabInterface(nlhs, plhs, nrhs, prhs);

#ifdef HAVE_PYTHON
			CPythonInterface::run_python_init();
#endif
#ifdef HAVE_OCTAVE
			COctaveInterface::run_octave_init();
#endif
#ifdef HAVE_R
			CRInterface::run_r_init();
#endif
		}
		else
			((CMatlabInterface*) interface)->reset(nlhs, plhs, nrhs, prhs);

		if (!interface->handle())
			SG_SERROR("Unknown command.\n");
	}
	catch (std::bad_alloc)
	{
		mexErrMsgTxt("Out of memory error.");
	}
	catch (ShogunException e)
	{
		mexErrMsgTxt(e.get_exception_string());
	}
	catch (...)
	{
		mexErrMsgTxt("Returning from SHOGUN in error.");
	}
}

/* to be run on exiting matlab ... does not seem to be possible right now:
 * run_octave_exit()
 * run_python_exit()
 * run_r_exit()
 */
