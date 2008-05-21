#include "lib/config.h"

#if defined(HAVE_R) && !defined(HAVE_SWIG)

#include "interface/RInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/r.h"

extern CSGInterface* interface;

CRInterface::CRInterface(SEXP prhs) : CSGInterface()
{
	reset(prhs);
}

CRInterface::~CRInterface()
{
}

void CRInterface::reset(SEXP prhs)
{
	CSGInterface::reset();

	m_nlhs=0;
	m_nrhs=length(prhs)-1;
	if (m_nrhs<0)
		m_nrhs=0;
	m_lhs=R_NilValue;
	m_rhs=prhs;
}


/** get functions - to pass data from the target interface to shogun */


/// get type of current argument (does not increment argument counter)
IFType CRInterface::get_argument_type()
{
	SEXP arg=CADR(m_rhs);

	switch (TYPEOF(arg))
	{
		case INTSXP:
			return DENSE_INT;
		case REALSXP:
			return DENSE_REAL;
		case STRSXP:
			return STRING_CHAR;
	};
	return UNDEFINED;
}


INT CRInterface::get_int()
{
	SEXP i=get_arg_increment();

	if (i == R_NilValue || nrows(CAR(i))!=1 || ncols(CAR(i))!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	if (TYPEOF(CAR(i)) == REALSXP)
	{
		double d=REAL(CAR(i))[0];
		if (d-CMath::floor(d)!=0)
			SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);
		return (INT) d;
	}

	if (TYPEOF(CAR(i)) != INTSXP)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return INTEGER(CAR(i))[0];
}

DREAL CRInterface::get_real()
{
	SEXP f=get_arg_increment();
	if (f == R_NilValue || TYPEOF(CAR(f)) != REALSXP || nrows(CAR(f))!=1 || ncols(CAR(f))!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return REAL(CAR(f))[0];
}

bool CRInterface::get_bool()
{
	SEXP b=get_arg_increment();
	if (b == R_NilValue || TYPEOF(CAR(b)) != LGLSXP || nrows(CAR(b))!=1 || ncols(CAR(b))!=1)
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

	SEXP rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Integer Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new INT[len];
	ASSERT(vec);

	for (INT i=0; i<len; i++)
		vec[i]= (INT) INTEGER(rvec)[i];
}

void CRInterface::get_shortreal_vector(SHORTREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_real_vector(DREAL*& vec, INT& len)
{
	SEXP rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != REALSXP && TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new DREAL[len];
	ASSERT(vec);

	for (INT i=0; i<len; i++)
		vec[i]= (DREAL) REAL(rvec)[i];
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
	SEXP feat=CAR(get_arg_increment());
	if( TYPEOF(feat) != REALSXP && TYPEOF(feat) != INTSXP )
		SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);

	num_vec = ncols(feat);
	num_feat = nrows(feat);
	matrix=new DREAL[num_vec*num_feat];
	ASSERT(matrix);

	for (INT i=0; i<num_vec; i++)
	{
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]= (DREAL) REAL(feat)[i*num_feat+j];
	}
}

void CRInterface::get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)
{
}


void CRInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void CRInterface::get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len)
{
}

void CRInterface::get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len)
{
	SEXP strs=get_arg_increment();

	if (strs == R_NilValue || TYPEOF(CAR(strs)) != STRSXP)
		SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);
	strs=CAR(strs);

	max_string_len=0;
	num_str=length(strs);
	strings=new T_STRING<CHAR>[num_str];
	ASSERT(strings);

	for (int i=0; i<num_str; i++)
	{
		SEXPREC* s= STRING_ELT(strs,i);
		CHAR* c= (CHAR*) CHAR(s);
		int len=LENGTH(s);

		if (len && c)
		{
			CHAR* dst=new CHAR[len+1];
			strings[i].string=(CHAR*) memcpy(dst, c, len*sizeof(CHAR));
			strings[i].string[len]='\0';
			strings[i].length=len;
			max_string_len=CMath::max(max_string_len, len);
		}
		else
		{
			SG_WARNING( "string with index %d has zero length\n", i+1);
			strings[i].string=0;
			strings[i].length=0;
		}
	}
}

void CRInterface::get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len)
{
}

void CRInterface::get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len)
{
}

void CRInterface::get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len)
{
}

/** set functions - to pass data from shogun to the target interface */
bool CRInterface::create_return_values(INT num)
{
	if (num<=0)
		return true;

	PROTECT(m_lhs=allocVector(VECSXP, num));
	m_nlhs=num;
	return length(m_lhs) == num;
}

SEXP CRInterface::get_return_values()
{
	if (m_nlhs>0)
		UNPROTECT(1);

	if (m_nlhs==1)
	{
		SEXP arg=VECTOR_ELT(m_lhs, 0);
		SET_VECTOR_ELT(m_lhs, m_lhs_counter, R_NilValue);
		return arg;
	}
	return m_lhs;
}


/** set functions - to pass data from shogun to the target interface */

void CRInterface::set_int(INT scalar)
{
	set_arg_increment(ScalarInteger(scalar));
}

void CRInterface::set_real(DREAL scalar)
{
	set_arg_increment(ScalarReal(scalar));
}

void CRInterface::set_bool(bool scalar)
{
	set_arg_increment(ScalarLogical(scalar));
}


void CRInterface::set_char_vector(const CHAR* vec, INT len)
{
}

#undef SET_VECTOR
#define SET_VECTOR(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CRInterface::function_name(const sg_type* vec, INT len)	\
{																\
	SEXP feat=NULL;												\
	PROTECT( feat = allocVector(r_type, len) );					\
																\
	for (INT i=0; i<len; i++)									\
		r_cast(feat)[i]=(if_type) vec[i];						\
																\
	UNPROTECT(1);												\
	set_arg_increment(feat);									\
}

SET_VECTOR(set_byte_vector, INTSXP, INTEGER, BYTE, int, "Byte")
SET_VECTOR(set_int_vector, INTSXP, INTEGER, INT, int, "Integer")
SET_VECTOR(set_short_vector, INTSXP, INTEGER, SHORT, int, "Short")
SET_VECTOR(set_shortreal_vector, REALSXP, REAL, SHORTREAL, float, "Single Precision")
SET_VECTOR(set_real_vector, REALSXP, REAL, DREAL, double, "Double Precision")
SET_VECTOR(set_word_vector, INTSXP, INTEGER, WORD, int, "Word")
#undef SET_VECTOR

void CRInterface::set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)
{
}

#define SET_MATRIX(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CRInterface::function_name(const sg_type* matrix, INT num_feat, INT num_vec) \
{																			\
	SEXP feat=NULL;															\
	PROTECT( feat = allocMatrix(r_type, num_feat, num_vec) );				\
																			\
	for (INT i=0; i<num_vec; i++)											\
	{																		\
		for (INT j=0; j<num_feat; j++)										\
			r_cast(feat)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];		\
	}																		\
																			\
	UNPROTECT(1);															\
	set_arg_increment(feat);												\
}
SET_MATRIX(set_byte_matrix, INTSXP, INTEGER, BYTE, int, "Byte")
SET_MATRIX(set_int_matrix, INTSXP, INTEGER, INT, int, "Integer")
SET_MATRIX(set_short_matrix, INTSXP, INTEGER, SHORT, int, "Short")
SET_MATRIX(set_shortreal_matrix, REALSXP, REAL, SHORTREAL, float, "Single Precision")
SET_MATRIX(set_real_matrix, REALSXP, REAL, DREAL, double, "Double Precision")
SET_MATRIX(set_word_matrix, INTSXP, INTEGER, WORD, int, "Word")
#undef SET_MATRIX

void CRInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec, LONG nnz)
{
	// R does not support sparse matrices yet
}

void CRInterface::set_byte_string_list(const T_STRING<BYTE>* strings, INT num_str)
{
}
 //this function will fail for strings containing 0, unclear how to do 'raw'
 //strings in R
void CRInterface::set_char_string_list(const T_STRING<CHAR>* strings, INT num_str)
{
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	SEXP feat=NULL;
	PROTECT( feat = allocVector(STRSXP, num_str) );

	for (INT i=0; i<num_str; i++)
	{
		INT len=strings[i].length;
		if (len>0)
			SET_STRING_ELT(feat, i, mkChar(strings[i].string));
	}
	UNPROTECT(1);
	set_arg_increment(feat);
}

void CRInterface::set_int_string_list(const T_STRING<INT>* strings, INT num_str)
{
}

void CRInterface::set_short_string_list(const T_STRING<SHORT>* strings, INT num_str)
{
}

void CRInterface::set_word_string_list(const T_STRING<WORD>* strings, INT num_str)
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

	try
	{
		if (!interface)
		{
			interface=new CRInterface(args);
			ASSERT(interface);
		}
		else
			((CRInterface*) interface)->reset(args);

		if (!interface->handle())
			SG_ERROR("Unknown command.\n");
	}
	catch (std::bad_alloc)
	{
		SG_PRINT("Out of memory error.\n");
		return R_NilValue;
	}
	catch (ShogunException e)
	{
		SG_PRINT("%s", e.get_exception_string());
		return R_NilValue;
	}

	return ((CRInterface*) interface)->get_return_values();
}

/* This method is called form within R when the current module is unregistered.
 * Note that R does not allow unregistering of single symbols. */

void R_unload_sg(DllInfo *info) { }

} // extern "C"

#endif // HAVE_R && !HAVE_SWIG
