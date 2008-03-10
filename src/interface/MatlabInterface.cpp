#include "lib/config.h"

#if defined(HAVE_MATLAB) && !defined(HAVE_SWIG)

#include "interface/MatlabInterface.h"
#include "interface/SGInterface.h"

#include "lib/io.h"
#include "lib/matlab.h"
#include "lib/ShogunException.h"
#include <mexversion.c>

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


void CMatlabInterface::get_byte_vector(BYTE*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 ||
		!(mxIsClass(mx_vec,"int8") || mxIsClass(mx_vec, "uint8")))
		SG_ERROR("Expected Byte Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new BYTE[len];
	ASSERT(vector);
	BYTE* data=(BYTE*) mxGetData(mx_vec);

	SG_DEBUG("BYTE vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_char_vector(CHAR*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 || !(mxIsChar(mx_vec)))
		SG_ERROR("Expected Char Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new CHAR[len];
	ASSERT(vector);
	CHAR* data=(CHAR*) mxGetData(mx_vec);

	SG_DEBUG("CHAR vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_int_vector(INT*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 ||
		!(
			mxIsClass(mx_vec,"int8") || mxIsClass(mx_vec, "int16") ||
			mxIsClass(mx_vec,"int32") || mxIsClass(mx_vec, "int64"))
	)
		SG_ERROR("Expected Integer Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new INT[len];
	ASSERT(vector);
	INT* data=(INT*) mxGetData(mx_vec);

	SG_DEBUG("INT vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_shortreal_vector(SHORTREAL*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 || !mxIsSingle(mx_vec))
		SG_ERROR(
			"Expected Single Precision Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new SHORTREAL[len];
	ASSERT(vector);
	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_vec);

	SG_DEBUG("SHORTREAL vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_real_vector(DREAL*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 || !mxIsDouble(mx_vec))
		SG_ERROR("Expected Double Precision Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new DREAL[len];
	ASSERT(vector);
	double* data=mxGetPr(mx_vec);

	SG_DEBUG("SHORTREAL vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_short_vector(SHORT*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 ||
		!(mxIsClass(mx_vec,"int16") || mxIsClass(mx_vec, "uint16"))
	)
		SG_ERROR("Expected Short Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new SHORT[len];
	ASSERT(vector);
	SHORT* data=(SHORT*) mxGetData(mx_vec);

	SG_DEBUG("SHORT vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}

void CMatlabInterface::get_word_vector(WORD*& vector, INT& len)
{
	const mxArray* mx_vec=get_arg_increment();
	if (!mx_vec || mxGetM(mx_vec)!=1 ||
		!(mxIsClass(mx_vec,"uint16"))
	)
		SG_ERROR("Expected Word Vector, got class %s as argument %d\n",
			mxGetClassName(mx_vec), m_rhs_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new WORD[len];
	ASSERT(vector);
	WORD* data=(WORD*) mxGetData(mx_vec);

	SG_DEBUG("WORD vector has %d elements.\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=data[i];
}


void CMatlabInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !(mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "uint8")))
		SG_ERROR("Expected Byte Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new BYTE[num_vec*num_feat];
	ASSERT(matrix);
	BYTE* data=(BYTE*) mxGetData(mx_mat);

	SG_DEBUG("dense BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsChar(mxGetCell(mx_mat, 1)))
		SG_ERROR("Expected Char Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new CHAR[num_vec*num_feat];
	ASSERT(matrix);
	CHAR* data=(CHAR*) mxGetData(mx_mat);

	SG_DEBUG("dense CHAR matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat ||
		!(
			mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "int16") ||
			mxIsClass(mx_mat,"int32") || mxIsClass(mx_mat, "int64"))
	)
		SG_ERROR("Expected Integer Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new INT[num_vec*num_feat];
	ASSERT(matrix);
	INT* data=(INT*) mxGetData(mx_mat);

	SG_DEBUG("dense INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSingle(mx_mat))
		SG_ERROR("Expected Single Precision Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new SHORTREAL[num_vec*num_feat];
	ASSERT(matrix);
	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_mat);

	SG_DEBUG("dense SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsDouble(mx_mat))
		SG_ERROR("Expected Double Precision Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new DREAL[num_vec*num_feat];
	ASSERT(matrix);
	DREAL* data=(DREAL*) mxGetData(mx_mat);

	SG_DEBUG("dense DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat ||
		!(mxIsClass(mx_mat,"uint16") || !mxIsClass(mx_mat, "int16"))
	)
		SG_ERROR("Expected Short Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new SHORT[num_vec*num_feat];
	ASSERT(matrix);
	SHORT* data=(SHORT*) mxGetData(mx_mat);

	SG_DEBUG("dense SHORT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}

void CMatlabInterface::get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsClass(mx_mat,"uint16"))
		SG_ERROR("Expected Integer Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new WORD[num_vec*num_feat];
	ASSERT(matrix);
	WORD* data=(WORD*) mxGetData(mx_mat);

	SG_DEBUG("dense WORD matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=data[i*num_feat+j];
}


void CMatlabInterface::get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!(mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "uint8")))
		SG_ERROR("Expected Byte Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<BYTE>[num_vec];
	ASSERT(matrix);
	BYTE* data=(BYTE*) mxGetData(mx_mat);

	SG_DEBUG("sparse BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<BYTE>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mxIsChar(mxGetCell(mx_mat, 1)))
		SG_ERROR("Expected Char Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<CHAR>[num_vec];
	ASSERT(matrix);
	CHAR* data=(CHAR*) mxGetData(mx_mat);

	SG_DEBUG("sparse CHAR matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<CHAR>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mx_mat || !(
		mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "int16") ||
		mxIsClass(mx_mat,"int32") || mxIsClass(mx_mat, "int64"))
	)
		SG_ERROR("Expected Integer Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<INT>[num_vec];
	ASSERT(matrix);
	INT* data=(INT*) mxGetData(mx_mat);

	SG_DEBUG("sparse INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<INT>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mxIsSingle(mx_mat))
		SG_ERROR("Expected Single Precision Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<SHORTREAL>[num_vec];
	ASSERT(matrix);
	SHORTREAL* data=(SHORTREAL*) mxGetData(mx_mat);

	SG_DEBUG("sparse SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<SHORTREAL>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mxIsDouble(mx_mat))
		SG_ERROR("Expected Double Precision Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<DREAL>[num_vec];
	ASSERT(matrix);
	DREAL* data=(DREAL*) mxGetData(mx_mat);

	SG_DEBUG("sparse DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<DREAL>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mx_mat ||
		!(mxIsClass(mx_mat,"uint16") || mxIsClass(mx_mat, "int16"))
	)
		SG_ERROR("Expected Integer Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<SHORT>[num_vec];
	ASSERT(matrix);
	SHORT* data=(SHORT*) mxGetData(mx_mat);

	SG_DEBUG("sparse SHORT matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<SHORT>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_arg_increment();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", m_rhs_counter);

	if (!mx_mat || !mxIsClass(mx_mat, "uint16"))
		SG_ERROR("Expected Integer Matrix, got class %s as argument %d\n",
			mxGetClassName(mx_mat), m_rhs_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<WORD>[num_vec];
	ASSERT(matrix);
	WORD* data=(WORD*) mxGetData(mx_mat);

	SG_DEBUG("sparse INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	LONG nzmax=mxGetNzmax(mx_mat);
	mwIndex* ir=mxGetIr(mx_mat);
	mwIndex* jc=mxGetJc(mx_mat);
	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=jc[i+1]-jc[i];
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<WORD>[len];
			ASSERT(matrix[i].features);

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=data[offset];
				matrix[i].features[j].feat_index=ir[offset];
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset==nzmax);
}

void CMatlabInterface::get_string_list(T_STRING<WORD>*& strings, INT& num_str)
{
	const mxArray* mx_str=get_arg_increment();
	if (!mx_str || !mxIsChar(mx_str))
		SG_ERROR("Expected String, got class %s as argument %d\n",
			mxGetClassName(mx_str), m_rhs_counter);

	mxChar* data=mxGetChars(mx_str);
	INT len=mxGetN(mx_str);
	num_str=mxGetM(mx_str);
	strings=new T_STRING<WORD>[num_str];
	ASSERT(strings);

	for (INT i=0; i<num_str; i++)
	{
		if (len>0)
		{
			strings[i].length=len; // all must have same length in matlab
			strings[i].string=new WORD[len+1]; // not zero terminated in matlab
			ASSERT(strings[i].string);
			INT j;
			for (j=0; j<len; j++)
				strings[i].string[j]=data[i+j*num_str];
			strings[i].string[j]='\0';
		}
		else
		{
			SG_WARNING( "string with index %d has zero length\n", i+1);
			strings[i].length=0;
			strings[i].string=NULL;
		}
	}
}

void CMatlabInterface::get_string_list(T_STRING<CHAR>*& strings, INT& num_str)
{
	const mxArray* mx_str=get_arg_increment();
	if (!mx_str || !mxIsChar(mx_str))
		SG_ERROR("Expected String, got class %s as argument %d\n",
			mxGetClassName(mx_str), m_rhs_counter);

	mxChar* data=mxGetChars(mx_str);
	INT len=mxGetN(mx_str);
	num_str=mxGetM(mx_str);
	strings=new T_STRING<CHAR>[num_str];
	ASSERT(strings);

	for (INT i=0; i<num_str; i++)
	{
		if (len>0)
		{
			strings[i].length=len; // all must have same length in matlab
			strings[i].string=new CHAR[len+1]; // not zero terminated in matlab
			ASSERT(strings[i].string);
			INT j;
			for (j=0; j<len; j++)
				strings[i].string[j]=data[i+j*num_str];
			strings[i].string[j]='\0';
		}
		else
		{
			SG_WARNING( "string with index %d has zero length\n", i+1);
			strings[i].length=0;
			strings[i].string=NULL;
		}
	}
}


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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
		SG_ERROR("Couldn't create Sparse Matrix of %d rows and %d cols\n", num_feat, num_vec);

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
