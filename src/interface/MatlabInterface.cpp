#include "lib/config.h"

#if defined(HAVE_MATLAB) && !defined(HAVE_SWIG)

#include <mexversion.c>

#include "interface/MatlabInterface.h"
#include "interface/SGInterface.h"

extern CSGInterface* interface;

CMatlabInterface::CMatlabInterface(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]) : CSGInterface()
{
	m_nlhs=nlhs;
	m_nrhs=nrhs;
	m_lhs=plhs;
	m_rhs=prhs;
}

CMatlabInterface::~CMatlabInterface()
{
}

/** get functions - to pass data from the target interface to shogun */
void CMatlabInterface::parse_args(INT num_args, INT num_default_args)
{
}


/// get type of current argument (does not increment argument counter)
IFType CMatlabInterface::get_argument_type()
{
	return UNDEFINED;
}


INT CMatlabInterface::get_int()
{
	const mxArray* i=get_arg_increment();
	if (!i || !mxIsNumeric(i) || mxGetN(i)!=1 || mxGetM(i)!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	double s=mxGetScalar(i);
	if (s-CMath::floor(s)!=0)
		SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);

	return INT(s);
}

DREAL CMatlabInterface::get_real()
{
	const mxArray* f=get_arg_increment();
	if (!f || !mxIsNumeric(f) || mxGetN(f)!=1 || mxGetM(f)!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return mxGetScalar(f);
}

bool CMatlabInterface::get_bool()
{
	const mxArray* b=get_arg_increment();
	if (!mxIsLogicalScalar(b))
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return *mxGetLogicals(b)==0;
}


CHAR* CMatlabInterface::get_string(INT& len)
{
	bool zero_terminate=true;
	const mxArray* s=get_arg_increment();

	if ( !(mxIsChar(s)) || (mxGetM(s)!=1) )
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	len=mxGetN(s);
	CHAR* string=NULL;
	if (zero_terminate)
		string=new CHAR[len+1];
	else
		string=new CHAR[len];
	ASSERT(string);
	mxChar* c=mxGetChars(s);
	ASSERT(c);
	for (INT i=0; i<len; i++)
		string[i]= (CHAR) (c[i]);

	if (zero_terminate)
		string[len]='\0';

	return string;
}

#define GET_VECTOR(function_name, mx_type, sg_type, if_type, error_string)	\
void CMatlabInterface::function_name(sg_type*& vector, INT& len)	\
{ 																	\
	const mxArray* mx_vec=get_arg_increment();						\
	if (!mx_vec || mxGetM(mx_vec)!=1 || !mxIsClass(mx_vec, mx_type))		\
		SG_ERROR("Expected " error_string " Vector, got class %s as argument %d\n", \
			mxGetClassName(mx_vec), m_rhs_counter); 				\
																	\
	len=mxGetNumberOfElements(mx_vec); 								\
	vector=new sg_type[len];										\
	ASSERT(vector);													\
	if_type* data=(if_type*) mxGetData(mx_vec);						\
																	\
	for (INT i=0; i<len; i++)										\
			vector[i]=data[i];										\
}

GET_VECTOR(get_byte_vector, "uint8", BYTE, BYTE, "Byte")
GET_VECTOR(get_char_vector, "char", CHAR, mxChar, "Char")
GET_VECTOR(get_int_vector, "int32", INT, int, "Integer")
GET_VECTOR(get_short_vector, "int16", SHORT, short, "Short")
GET_VECTOR(get_shortreal_vector, "single", SHORTREAL, float, "Single Precision")
GET_VECTOR(get_real_vector, "double", DREAL, double, "Double Precision")
GET_VECTOR(get_word_vector, "uint16", WORD, unsigned short, "Word")
#undef GET_VECTOR

#define GET_MATRIX(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(sg_type*& matrix, INT& num_feat, INT& num_vec) \
{ 																				\
	const mxArray* mx_mat=get_arg_increment(); 									\
	if (!mx_mat || !(mxIsClass(mx_mat, mx_type))) 								\
		SG_ERROR("Expected " error_string " Matrix, got class %s as argument %d\n", \
			mxGetClassName(mx_mat), m_rhs_counter); 							\
 																				\
	num_vec=mxGetN(mx_mat); 													\
	num_feat=mxGetM(mx_mat); 													\
	matrix=new sg_type[num_vec*num_feat]; 										\
	ASSERT(matrix); 															\
	if_type* data=(if_type*) mxGetData(mx_mat); 								\
 																				\
	for (INT i=0; i<num_vec; i++) 												\
		for (INT j=0; j<num_feat; j++) 											\
			matrix[i*num_feat+j]=data[i*num_feat+j];							\
}

GET_MATRIX(get_byte_matrix, "uint8", BYTE, BYTE, "Byte")
GET_MATRIX(get_char_matrix, "char", CHAR, mxChar, "Char")
GET_MATRIX(get_int_matrix, "int32", INT, int, "Integer")
GET_MATRIX(get_short_matrix, "int16", SHORT, short, "Short")
GET_MATRIX(get_shortreal_matrix, "single", SHORTREAL, float, "Single Precision")
GET_MATRIX(get_real_matrix, "double", DREAL, double, "Double Precision")
GET_MATRIX(get_word_matrix, "uint16", WORD, unsigned short, "Word")
#undef GET_MATRIX

#define GET_SPARSEMATRIX(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::function_name(TSparse<sg_type>*& matrix, INT& num_feat, INT& num_vec) \
{																						\
	const mxArray* mx_mat=get_arg_increment(); 											\
	if (!mx_mat || !mxIsSparse(mx_mat)) 												\
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter); 			\
 																						\
	if (!mxIsClass(mx_mat,mx_type)) 													\
		SG_ERROR("Expected " error_string " Matrix, got class %s as argument %d\n",	\
			mxGetClassName(mx_mat), m_rhs_counter); 									\
 																						\
	num_vec=mxGetN(mx_mat); 															\
	num_feat=mxGetM(mx_mat); 															\
	matrix=new TSparse<sg_type>[num_vec]; 												\
	ASSERT(matrix); 																	\
	if_type* data=(if_type*) mxGetData(mx_mat); 										\
 																						\
	LONG nzmax=mxGetNzmax(mx_mat); 														\
	mwIndex* ir=mxGetIr(mx_mat); 														\
	mwIndex* jc=mxGetJc(mx_mat); 														\
	LONG offset=0; 																		\
	for (INT i=0; i<num_vec; i++) 														\
	{ 																					\
		INT len=jc[i+1]-jc[i]; 															\
		matrix[i].vec_index=i; 															\
		matrix[i].num_feat_entries=len; 												\
 																						\
		if (len>0) 																		\
		{ 																				\
			matrix[i].features=new TSparseEntry<sg_type>[len]; 							\
			ASSERT(matrix[i].features); 												\
 																						\
			for (INT j=0; j<len; j++) 													\
			{ 																			\
				matrix[i].features[j].entry=data[offset]; 								\
				matrix[i].features[j].feat_index=ir[offset]; 							\
				offset++; 																\
			} 																			\
		} 																				\
		else 																			\
			matrix[i].features=NULL; 													\
	} 																					\
	ASSERT(offset==nzmax); 																\
}

GET_SPARSEMATRIX(get_byte_sparsematrix, "uint8", BYTE, BYTE, "Byte")
GET_SPARSEMATRIX(get_char_sparsematrix, "char", CHAR, mxChar, "Char")
GET_SPARSEMATRIX(get_int_sparsematrix, "int32", INT, int, "Integer")
GET_SPARSEMATRIX(get_short_sparsematrix, "int16", SHORT, short, "Short")
GET_SPARSEMATRIX(get_shortreal_sparsematrix, "single", SHORTREAL, float, "Single Precision")
GET_SPARSEMATRIX(get_real_sparsematrix, "double", DREAL, double, "Double Precision")
GET_SPARSEMATRIX(get_word_sparsematrix, "uint16", WORD, unsigned short, "Word")
#undef GET_SPARSEMATRIX


#define GET_STRINGLIST(function_name, mx_type, sg_type, if_type, error_string)		\
void CMatlabInterface::get_string_list(T_STRING<sg_type>*& strings, INT& num_str, INT& max_string_len) 	\
{ 																						\
	const mxArray* mx_str=get_arg_increment();											\
	if (!mx_str)																		\
		SG_ERROR("Expected Stringlist as argument (none given)\n");						\
																						\
	if (mxIsCell(mx_str))																\
	{																					\
		num_str=mxGetNumberOfElements(mx_str);											\
		ASSERT(num_str>=1);																\
																						\
		strings=new T_STRING<sg_type>[num_str];											\
		ASSERT(strings);																\
																						\
		for (int i=0; i<num_str; i++)													\
		{																				\
			mxArray* str=mxGetCell(mx_str, i);											\
			if (!str || !mxIsClass(str, mx_type) || !mxGetM(str)==1)							\
				SG_ERROR("Expected String of type " error_string " as argument %d\n", m_rhs_counter); \
																						\
			INT len=mxGetN(str);														\
			if (len>0) 																	\
			{ 																			\
				if_type* data=(if_type*) mxGetData(str);								\
				strings[i].length=len; /* all must have same length in matlab */ 		\
				strings[i].string=new sg_type[len+1]; /* not zero terminated in matlab */ \
				ASSERT(strings[i].string); 												\
				INT j; 																	\
				for (j=0; j<len; j++) 													\
					strings[i].string[j]=data[j]; 										\
				strings[i].string[j]='\0'; 												\
				max_string_len=CMath::max(max_string_len, len);							\
			}																			\
			else																		\
			{																			\
				SG_WARNING( "string with index %d has zero length\n", i+1);				\
				strings[i].length=0;													\
				strings[i].string=NULL;													\
			}																			\
		}																				\
	}																					\
	else if (mxIsClass(mx_str, mx_type))												\
	{																					\
		if_type* data=(if_type*) mxGetData(mx_str);										\
		INT len=mxGetN(mx_str); 														\
		num_str=mxGetM(mx_str); 														\
		strings=new T_STRING<sg_type>[num_str]; 										\
		ASSERT(strings); 																\
																						\
		for (INT i=0; i<num_str; i++) 													\
		{ 																				\
			if (len>0) 																	\
			{ 																			\
				strings[i].length=len; /* all must have same length in matlab */ 		\
				strings[i].string=new sg_type[len+1]; /* not zero terminated in matlab */ \
				ASSERT(strings[i].string); 												\
				INT j; 																	\
				for (j=0; j<len; j++) 													\
					strings[i].string[j]=data[i+j*num_str]; 							\
				strings[i].string[j]='\0'; 												\
			} 																			\
			else 																		\
			{ 																			\
				SG_WARNING( "string with index %d has zero length\n", i+1); 			\
				strings[i].length=0; 													\
				strings[i].string=NULL; 												\
			} 																			\
		} 																				\
		max_string_len=len;																\
	}																					\
	else																				\
		SG_ERROR("Expected String, got class %s as argument %d\n",						\
			mxGetClassName(mx_str), m_rhs_counter);										\
}

GET_STRINGLIST(get_byte_string_list, "uint8", BYTE, BYTE, "Byte")
GET_STRINGLIST(get_char_string_list, "char", CHAR, mxChar, "Char")
GET_STRINGLIST(get_int_string_list, "int32", INT, int, "Integer")
GET_STRINGLIST(get_short_string_list, "int16", SHORT, short, "Short")
GET_STRINGLIST(get_word_string_list, "uint16", WORD, unsigned short, "Word")


/** set functions - to pass data from shogun to the target interface */
void CMatlabInterface::create_return_values(INT num_val)
{
}

void CMatlabInterface::set_byte_vector(const BYTE* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxINT8_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Byte Vector of length %d\n", len);

	BYTE* data=(BYTE*) mxGetData(mx_vec);

	SG_DEBUG("BYTE vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_char_vector(const CHAR* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxCHAR_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Char Vector of length %d\n", len);

	CHAR* data=(CHAR*) mxGetData(mx_vec);

	SG_DEBUG("CHAR vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_int_vector(const INT* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxINT32_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Integer Vector of length %d\n", len);

	INT* data=(INT*) mxGetData(mx_vec);

	SG_DEBUG("INT vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_shortreal_vector(const SHORTREAL* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxSINGLE_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Single Precision Vector of length %d\n", len);

	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_vec);

	SG_DEBUG("SHORTREAL vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_real_vector(const DREAL* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxDOUBLE_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Double Precision Vector of length %d\n", len);

	DREAL* data=(DREAL*) mxGetData(mx_vec);

	SG_DEBUG("DREAL vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_short_vector(const SHORT* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxINT16_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Short Vector of length %d\n", len);

	SHORT* data=(SHORT*) mxGetData(mx_vec);

	SG_DEBUG("SHORT vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}

void CMatlabInterface::set_word_vector(const WORD* vector, INT len)
{
	if (!vector)
		SG_ERROR("Given vector is invalid\n");

	mxArray* mx_vec=mxCreateNumericMatrix(1, len, mxUINT16_CLASS, mxREAL);
	if (!mx_vec)
		SG_ERROR("Couldn't create Word Vector of length %d\n", len);

	WORD* data=(WORD*) mxGetData(mx_vec);

	SG_DEBUG("WORD vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
		data[i]=vector[i];

	set_arg_increment(mx_vec);
}


void CMatlabInterface::set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxINT8_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Byte Matrix of %d rows and %d cols\n", num_feat, num_vec);

	BYTE* data=(BYTE*) mxGetData(mx_mat);

	SG_DEBUG("dense BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxCHAR_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Char Matrix of %d rows and %d cols\n", num_feat, num_vec);

	CHAR* data=(CHAR*) mxGetData(mx_mat);

	SG_DEBUG("dense CHAR matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxINT32_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Integer Matrix of %d rows and %d cols\n", num_feat, num_vec);

	INT* data=(INT*) mxGetData(mx_mat);

	SG_DEBUG("dense INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxSINGLE_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Single Precision Matrix of %d rows and %d cols\n", num_feat, num_vec);

	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_mat);

	SG_DEBUG("dense SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxDOUBLE_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Double Precision Matrix of %d rows and %d cols\n", num_feat, num_vec);

	DREAL* data=(DREAL*) mxGetData(mx_mat);

	SG_DEBUG("dense DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxINT16_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Short Matrix of %d rows and %d cols\n", num_feat, num_vec);

	SHORT* data=(SHORT*) mxGetData(mx_mat);

	SG_DEBUG("dense SHORT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateNumericMatrix(num_feat, num_vec, mxUINT16_CLASS, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Word Matrix of %d rows and %d cols\n", num_feat, num_vec);

	WORD* data=(WORD*) mxGetData(mx_mat);

	SG_DEBUG("dense WORD matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			data[i*num_feat+j]=matrix[i*num_feat+j];

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	BYTE* data=(BYTE*) mxGetData(mx_mat);

	SG_DEBUG("sparse BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	CHAR* data=(CHAR*) mxGetData(mx_mat);

	SG_DEBUG("sparse CHAR matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	INT* data=(INT*) mxGetData(mx_mat);

	SG_DEBUG("sparse INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_mat);

	SG_DEBUG("sparse SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	DREAL* data=(DREAL*) mxGetData(mx_mat);

	SG_DEBUG("sparse DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	SHORT* data=(SHORT*) mxGetData(mx_mat);

	SG_DEBUG("sparse SHORT matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec)
{
	if (!matrix)
		SG_ERROR("Given matrix is invalid\n");

	mxArray* mx_mat=mxCreateSparse(num_feat, num_vec, num_feat*num_vec, mxREAL);
	if (!mx_mat)
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols.\n", num_feat, num_vec);

	WORD* data=(WORD*) mxGetData(mx_mat);

	SG_DEBUG("sparse WORD matrix has %d rows, %d cols\n", num_feat, num_vec);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		jc[i]=offset;
		for (INT j=0; j<len; j++)
		{
			data[offset]=matrix[i].features[j].entry;
			ir[offset]=matrix[i].features[j].feat_index;
			offset++;
		}
	}
	jc[num_vec]=offset;

	set_arg_increment(mx_mat);
}

void CMatlabInterface::set_string_list(const T_STRING<CHAR>* strings, INT num_str)
{
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	const CHAR* list[num_str];
	for (INT i=0; i<num_str; i++)
		list[i]=strings[i].string;

	mxArray* mx_str=mxCreateCharMatrixFromStrings(num_str, list);
	if (!mx_str)
		SG_ERROR("Couldn't create String Matrix of %d strings.\n", num_str);

	set_arg_increment(mx_str);
}

void CMatlabInterface::set_string_list(const T_STRING<WORD>* strings, INT num_str)
{
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	const CHAR* list[num_str];
	for (INT i=0; i<num_str; i++)
		list[i]=(CHAR*) strings[i].string;

	mxArray* mx_str=mxCreateCharMatrixFromStrings(num_str, list);
	if (!mx_str)
		SG_ERROR("Couldn't create String Matrix of %d strings.\n", num_str);

	set_arg_increment(mx_str);
}

void CMatlabInterface::submit_return_values()
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

void obsolete_mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[]);

void mexFunction(int nlhs,mxArray *plhs[],int nrhs,const mxArray *prhs[])
{
	delete interface;
	interface=new CMatlabInterface(nlhs, plhs, nrhs, prhs);
	if (!interface->handle())
	{
		SG_WARNING("falling back to obsolete interface\n");
		obsolete_mexFunction(nlhs, plhs, nrhs, prhs);
	}
}
#endif // HAVE_MATLAB && !HAVE_SWIG
