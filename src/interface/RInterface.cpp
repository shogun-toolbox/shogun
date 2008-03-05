#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)

#include "interface/RInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"

extern "C" {
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
}

extern CSGInterface* interface;

CRInterface::CRInterface(SEXP prhs) : CSGInterface()
{
	m_nlhs=0;
	m_nrhs=length(prhs);
	m_lhs=R_NilValue;
	m_rhs=prhs;
}

CRInterface::~CRInterface()
{
}

/** get functions - to pass data from the target interface to shogun */
void CRInterface::parse_args(INT num_args, INT num_default_args)
{
}


/// get type of current argument (does not increment argument counter)
IFType CRInterface::get_argument_type()
{
	return UNDEFINED;
}


INT CRInterface::get_int()
{
	SEXP i=get_arg_increment();
	if (i == R_NilValue || TYPEOF(CAR(i)) != INTSXP || Rf_nrows(CAR(i))!=1 || Rf_ncols(CAR(i))!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return INTEGER(CAR(i))[0];
}

DREAL CRInterface::get_real()
{
	SEXP f=get_arg_increment();
	if (f == R_NilValue || TYPEOF(CAR(f)) != REALSXP || Rf_nrows(CAR(f))!=1 || Rf_ncols(CAR(f))!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return REAL(CAR(f))[0];
}

bool CRInterface::get_bool()
{
	SEXP b=get_arg_increment();
	if (b == R_NilValue || TYPEOF(CAR(b)) != LGLSXP || Rf_nrows(CAR(b))!=1 || Rf_ncols(CAR(b))!=1)
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return INTEGER(CAR(b))[0] != 0;
}


CHAR* CRInterface::get_string(INT& len)
{
	SEXP s=get_arg_increment();
	if (s == R_NilValue || TYPEOF(CAR(s)) != STRSXP || length(CAR(s))!=1)
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	SEXPREC* rstr= STRING_ELT(CAR(s),0);
	const CHAR* str= CHAR(rstr);
	len=LENGTH(rstr);
	ASSERT(len>0);
	CHAR* res=new CHAR[len+1];
	memcpy(res, str, len*sizeof(CHAR));
	res[len]='\0';
	return res;
}

void CRInterface::get_byte_vector(BYTE*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_char_vector(CHAR*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_int_vector(INT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_shortreal_vector(SHORTREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_real_vector(DREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_short_vector(SHORT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_word_vector(WORD*& vec, INT& len)
{
	vec=NULL;
	len=0;
}


void CRInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	SEXP feat=get_arg_increment();
	if( TYPEOF(feat) != REALSXP && TYPEOF(feat) != INTSXP )
		SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);

	num_vec = Rf_ncols(feat);
	num_feat = Rf_nrows(feat);
	matrix=new DREAL[num_vec*num_feat];
	ASSERT(matrix);

	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]= (DREAL) REAL(feat)[i*num_feat+j];
}

void CRInterface::get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)
{
}


void CRInterface::get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec)
{
}


void CRInterface::get_string_list(T_STRING<CHAR>*& strings, INT& num_str)
{
	SEXP strs=get_arg_increment();
	if (strs == R_NilValue || TYPEOF(CAR(strs)) != STRSXP || length(CAR(strs))>=1)
		SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);

	num_str=length(CAR(strs));
	strings=new T_STRING<CHAR>[num_str];
	ASSERT(strings);

	for (int i=0; i<num_str; i++)
	{
		SEXPREC* s= STRING_ELT(strs,i);
		CHAR* c= (CHAR*) CHAR(s);
		int len=LENGTH(s);

		if (len && c)
		{
			CHAR* dst=new CHAR[len];
			strings[i].string=(CHAR*) memcpy(dst, c, len*sizeof(CHAR));
			strings[i].length=len;
		}
		else
		{
			SG_WARNING( "string with index %d has zero length\n", i+1);
			strings[i].string=0;
			strings[i].length=0;
		}
	}
}


/** set functions - to pass data from shogun to the target interface */
void CRInterface::create_return_values(INT num_val)
{
}

void CRInterface::set_byte_vector(const BYTE* vec, INT len)
{
}

void CRInterface::set_char_vector(const CHAR* vec, INT len)
{
}

void CRInterface::set_int_vector(const INT* vec, INT len)
{
}

void CRInterface::set_shortreal_vector(const SHORTREAL* vec, INT len)
{
}

void CRInterface::set_real_vector(const DREAL* vec, INT len)
{
}

void CRInterface::set_short_vector(const SHORT* vec, INT len)
{
}

void CRInterface::set_word_vector(const WORD* vec, INT len)
{
}


void CRInterface::set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)
{
}


void CRInterface::set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec)
{
}

void CRInterface::set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec)
{
}


void CRInterface::set_string_list(const T_STRING<CHAR>* strings, INT num_str)
{
}


void CRInterface::submit_return_values()
{
}

/* The main function of the shogun R interface. All commands from the R command line
 * to the shogun backend are passed using the syntax:
 * .External("sg", "func", ... ) 
 * where '...' is a number of arguments passed to the shogun function 'func'. */

extern "C" {
/* This method is called by R when the shogun module is loaded into R
 * via dyn.load('sg.so'). */

SEXP sg(SEXP args);

void R_init_sg(DllInfo *info) { 
   
   /* There are four different external language call mechanisms available in R, namely:
    *    .C
    *    .Call
    *    .Fortran
    *    .External
    *
    * Currently shogun uses only the .External interface. */

   R_CMethodDef cMethods[] = { {NULL, NULL, 0} };
   R_FortranMethodDef fortranMethods[] = { {NULL, NULL, 0} };
   R_ExternalMethodDef externalMethods[] = { {"sg", (void*(*)()) &sg, 1}, {NULL, NULL, 0} };
   R_CallMethodDef callMethods[] = { {NULL, NULL, 0} };

   /* Register the routines saved in the callMethods structure so that they are available under R. */
   R_registerRoutines(info, cMethods, callMethods, (R_FortranMethodDef*) fortranMethods, (R_ExternalMethodDef*) externalMethods);

}

SEXP sg(SEXP args)
{
	/* The SEXP (Simple Expression) args is a list of arguments of the .External call. 
	 * it consists of "sg", "func" and additional arguments.
	 * */

	delete interface;
	interface=new CRInterface(args);

	try
	{
		if (!interface->handle())
			SG_ERROR("interface currently does not handle this command\n");
	}
	catch (ShogunException e)
	{
		return R_NilValue;
	}

	return ((CRInterface*) interface)->get_return_values();
}

/* This method is called form within R when the current module is unregistered.
 * Note that R does not allow unregistering of single symbols. */

void R_unload_sg(DllInfo *info) { }

} // extern "C"

#endif // HAVE_R && !HAVE_SWIG
