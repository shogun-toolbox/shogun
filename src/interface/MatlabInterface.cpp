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

	return INT(s);
}

DREAL CMatlabInterface::get_real()
{
	const mxArray* f=get_current_arg();
	if (!f || !mxIsNumeric(f) || mxGetN(f)!=1 || mxGetM(f)!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", arg_counter);

	return mxGetScalar(f);
}

bool CMatlabInterface::get_bool()
{
	const mxArray* b=get_current_arg();
	if (!mxIsLogicalScalar(b))
		SG_ERROR("Expected Scalar Boolean as argument %d\n", arg_counter);

	return *mxGetLogicals(b)==0;
}


CHAR* CMatlabInterface::get_string(INT& len)
{
	bool zero_terminate=true;
	const mxArray* s=get_current_arg();

	if ( (mxIsChar(s)) && (mxGetM(s)==1) )
	{
		len = mxGetN(s);
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
	else
		throw ShogunException("Expecting string as argument"); //TODO print out arg nr?
}


INT CMatlabInterface::get_int_from_string()
{
	return 42;
}

DREAL CMatlabInterface::get_real_from_string()
{
	return 42;
}

bool CMatlabInterface::get_bool_from_string()
{
	return false;
}


void CMatlabInterface::get_byte_vector(BYTE** vec, INT* len)
{
	*vec=NULL;
	*len=0;
}

void CMatlabInterface::get_int_vector(INT** vec, INT* len)
{
	*vec=NULL;
	*len=0;
}

void CMatlabInterface::get_shortreal_vector(SHORTREAL** vec, INT* len)
{
	*vec=NULL;
	*len=0;
}

void CMatlabInterface::get_real_vector(DREAL** vec, INT* len)
{
	*vec=NULL;
	*len=0;
}


void CMatlabInterface::get_byte_matrix(BYTE** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_int_matrix(INT** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_shortreal_matrix(SHORTREAL** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_real_matrix(DREAL** matrix, INT* num_feat, INT* num_vec)
{
	const mxArray* mx_mat=get_current_arg();
	if (!mxIsDouble(mx_mat))
		SG_ERROR("Expected Double Matrix as argument %d\n", arg_counter);

	*num_vec=mxGetN(mx_mat);
	*num_feat=mxGetM(mx_mat);
	INT nf=*num_feat;
	INT nv=*num_vec;
	*matrix=new DREAL[nv*nf];
	DREAL* mat=*matrix;
	ASSERT(mat);
	double* feat=mxGetPr(mx_mat);

	SG_DEBUG("dense matrix has %d rows, %d cols\n", nf, nv);
	for (INT i=0; i<nv; i++)
		for (INT j=0; j<nf; j++)
			mat[i*nf+j]=feat[i*nf+j];
}


void CMatlabInterface::get_byte_sparsematrix(TSparse<BYTE>** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_int_sparsematrix(TSparse<INT>** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>** matrix, INT* num_feat, INT* num_vec)
{
}

void CMatlabInterface::get_real_sparsematrix(TSparse<DREAL>** matrix, INT* num_feat, INT* num_vec)
{
}


void CMatlabInterface::get_string_list(T_STRING<CHAR>** strings, INT* num_str)
{
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
