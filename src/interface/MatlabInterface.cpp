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
	const mxArray* i=get_current_arg();
	if (!i || !mxIsNumeric(i) || mxGetN(i)!=1 || mxGetM(i)!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", arg_counter);

	double s=mxGetScalar(i);
	if (s-CMath::floor(s)!=0)
		SG_ERROR("Expected Integer as argument %d\n", arg_counter);

	arg_counter++;
	return INT(s);
}

DREAL CMatlabInterface::get_real()
{
	const mxArray* f=get_current_arg();
	if (!f || !mxIsNumeric(f) || mxGetN(f)!=1 || mxGetM(f)!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", arg_counter);

	arg_counter++;
	return mxGetScalar(f);
}

bool CMatlabInterface::get_bool()
{
	const mxArray* b=get_current_arg();
	if (!mxIsLogicalScalar(b))
		SG_ERROR("Expected Scalar Boolean as argument %d\n", arg_counter);

	arg_counter++;
	return *mxGetLogicals(b)==0;
}


CHAR* CMatlabInterface::get_string(INT& len)
{
	bool zero_terminate=true;
	const mxArray* s=get_current_arg();

	if ( !(mxIsChar(s)) || (mxGetM(s)!=1) )
		SG_ERROR("Expected String as argument %d\n", arg_counter);

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

	arg_counter++;
	return string;
}


void CMatlabInterface::get_byte_vector(BYTE*& vector, INT& len)
{
	const mxArray* mx_vec=get_current_arg();
	if (!mx_vec || mxGetN(mx_vec)!=1 ||
		!(mxIsClass(mx_vec,"int8") || mxIsClass(mx_vec, "uint8")))
		SG_ERROR("Expected Byte Vector as argument %d\n", arg_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new BYTE[len];
	ASSERT(vector);
	BYTE* feat=(BYTE*) mxGetData(mx_vec);

	SG_DEBUG("BYTE vector has %d elements\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=feat[i];

	arg_counter++;
}

void CMatlabInterface::get_int_vector(INT*& vector, INT& len)
{
	const mxArray* mx_vec=get_current_arg();
	if (!mx_vec || mxGetN(mx_vec)!=1 || !(
		mxIsClass(mx_vec,"int8") || mxIsClass(mx_vec, "int16") ||
		mxIsClass(mx_vec,"int32") || mxIsClass(mx_vec, "int64"))
	)
		SG_ERROR("Expected Integer Vector as argument %d\n", arg_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new INT[len];
	ASSERT(vector);
	INT* feat=(INT*) mxGetData(mx_vec);

	SG_DEBUG("INT vector has %d elements\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=feat[i];

	arg_counter++;
}

void CMatlabInterface::get_shortreal_vector(SHORTREAL*& vector, INT& len)
{
	const mxArray* mx_vec=get_current_arg();
	if (!mx_vec || mxGetN(mx_vec)!=1 || !mxIsSingle(mx_vec))
		SG_ERROR("Expected Single Precision Vector as argument %d\n", arg_counter);

	len=mxGetM(mx_vec);
	vector=new SHORTREAL[len];
	ASSERT(vector);
	SHORTREAL* feat=(SHORTREAL*) mxGetData(mx_vec);

	SG_DEBUG("SHORTREAL vector has %d elements\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=feat[i];

	arg_counter++;
}

void CMatlabInterface::get_real_vector(DREAL*& vector, INT& len)
{
	const mxArray* mx_vec=get_current_arg();
	if (!mx_vec || mxGetN(mx_vec)!=1 || !mxIsDouble(mx_vec))
		SG_ERROR("Expected Double Precision Vector as argument %d\n", arg_counter);

	len=mxGetNumberOfElements(mx_vec);
	vector=new DREAL[len];
	ASSERT(vector);
	double* feat=mxGetPr(mx_vec);

	SG_DEBUG("SHORTREAL vector has %d elements\n", len);
	for (INT i=0; i<len; i++)
			vector[i]=feat[i];

	arg_counter++;
}


void CMatlabInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !(mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "uint8")))
		SG_ERROR("Expected Byte Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new BYTE[num_vec*num_feat];
	ASSERT(matrix);
	BYTE* feat=(BYTE*) mxGetData(mx_mat);

	SG_DEBUG("dense BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !(
		mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "int16") ||
		mxIsClass(mx_mat,"int32") || mxIsClass(mx_mat, "int64"))
	)
		SG_ERROR("Expected Integer Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new INT[num_vec*num_feat];
	ASSERT(matrix);
	INT* feat=(INT*) mxGetData(mx_mat);

	SG_DEBUG("dense INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsSingle(mx_mat))
		SG_ERROR("Expected Single Precision Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new SHORTREAL[num_vec*num_feat];
	ASSERT(matrix);
	SHORTREAL* feat=(SHORTREAL*) mxGetData(mx_mat);

	SG_DEBUG("dense SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsDouble(mx_mat))
		SG_ERROR("Expected Double Precision Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new DREAL[num_vec*num_feat];
	ASSERT(matrix);
	double* feat=mxGetPr(mx_mat);

	SG_DEBUG("dense DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}


void CMatlabInterface::get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", arg_counter);

	if (!(mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "uint8")))
		SG_ERROR("Expected Byte Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<BYTE>[num_vec*num_feat];
	ASSERT(matrix);
	TSparse<BYTE>* feat=(TSparse<BYTE>*) mxGetData(mx_mat);

	SG_DEBUG("sparse BYTE matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", arg_counter);

	if (!mx_mat || !(
		mxIsClass(mx_mat,"int8") || mxIsClass(mx_mat, "int16") ||
		mxIsClass(mx_mat,"int32") || mxIsClass(mx_mat, "int64"))
	)
		SG_ERROR("Expected Integer Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<INT>[num_vec*num_feat];
	ASSERT(matrix);
	TSparse<INT>* feat=(TSparse<INT>*) mxGetData(mx_mat);

	SG_DEBUG("sparse INT matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", arg_counter);

	if (!mxIsSingle(mx_mat))
		SG_ERROR("Expected Single Precision Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<SHORTREAL>[num_vec*num_feat];
	ASSERT(matrix);
	TSparse<SHORTREAL>* feat=(TSparse<SHORTREAL>*) mxGetData(mx_mat);

	SG_DEBUG("sparse SHORTREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}

void CMatlabInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mx_mat || !mxIsSparse(mx_mat))
		SG_ERROR("Expected Sparse Matrix as argument %d\n", arg_counter);

	if (!mxIsDouble(mx_mat))
		SG_ERROR("Expected Double Precision Matrix as argument %d\n", arg_counter);

	num_vec=mxGetN(mx_mat);
	num_feat=mxGetM(mx_mat);
	matrix=new TSparse<DREAL>[num_vec*num_feat];
	ASSERT(matrix);
	TSparse<DREAL>* feat=(TSparse<DREAL>*) mxGetData(mx_mat);

	SG_DEBUG("sparse DREAL matrix has %d rows, %d cols\n", num_feat, num_vec);
	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]=feat[i*num_feat+j];

	arg_counter++;
}


void CMatlabInterface::get_string_list(T_STRING<CHAR>*& strings, INT& num_str)
{
	const mxArray* mx_str=get_current_arg();
	if (!mx_str || !mxIsChar(mxGetCell(mx_str, 0)))
		SG_ERROR("Expected String List as argument %d\n", arg_counter);

	num_str=mxGetNumberOfElements(mx_str);
	strings=new T_STRING<CHAR>[num_str];

	for (INT i=0; i<num_str; i++)
	{
		mxArray* str=mxGetCell(mx_str, i);
		if (!str || !mxIsChar(str))
			SG_ERROR("Expected String as argument %d, index %d\n", arg_counter, i);

		INT len=(INT) mxGetElementSize(str);
		if (len>0)
		{
			CHAR* dst=new CHAR[len];
			strings[i].string=(CHAR*) memcpy(dst, mxGetChars(str), len*sizeof(CHAR));
			strings[i].length=len;
		}
		else
		{
			SG_WARNING( "string with index %d has zero length\n", i+1);
			strings[i].string=0;
			strings[i].length=0;
		}
	}

	arg_counter++;
}


/** set functions - to pass data from shogun to the target interface */
void CMatlabInterface::create_return_values(INT num_val)
{
}

void CMatlabInterface::set_byte_vector(BYTE* vec, INT len)
{
}

void CMatlabInterface::set_int_vector(INT* vec, INT len)
{
}

void CMatlabInterface::set_shortreal_vector(SHORTREAL* vec, INT len)
{
}

void CMatlabInterface::set_real_vector(DREAL* vec, INT len)
{
}


void CMatlabInterface::set_byte_matrix(BYTE* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_int_matrix(INT* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_shortreal_matrix(SHORTREAL* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_real_matrix(DREAL* matrix, INT num_feat, INT num_vec)
{
}


void CMatlabInterface::set_byte_sparsematrix(TSparse<BYTE>* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_int_sparsematrix(TSparse<INT>* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_shortreal_sparsematrix(TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)
{
}

void CMatlabInterface::set_real_sparsematrix(TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
}


void CMatlabInterface::set_string_list(T_STRING<CHAR>* strings, INT num_str)
{
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
#endif // HAVE_MATLAB && HAVE_SWIG
