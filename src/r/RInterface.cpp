#include "RInterface.h"

#include <stdio.h>
#include <shogun/ui/SGInterface.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/lib/io.h>
#include <shogun/base/init.h>


#ifdef HAVE_PYTHON
#include <dlfcn.h>
#include "../python/PythonInterface.h"
#endif

void r_print_message(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void r_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		Rprintf((char*) "%s", str);
	else
		fprintf(target, "%s", str);
}

void r_print_error(FILE* target, const char* str)
{
	if (target!=stdout)
		fprintf(target, "%s", str);
}

void r_cancel_computations(bool &delayed, bool &immediately)
{
			//R_Suicide((char*) "sg stopped by SIGINT\n");
}

extern CSGInterface* interface;

CRInterface::CRInterface(SEXP prhs)
: CSGInterface()
{
	reset(prhs);

#ifdef HAVE_PYTHON
	m_pylib = dlopen(LIBPYTHON, RTLD_NOW | RTLD_GLOBAL);
	if (!m_pylib)
		SG_ERROR("couldn't open " LIBPYTHON ".so\n");
	Py_Initialize();
	import_array();
#endif
}

CRInterface::~CRInterface()
{
#ifdef HAVE_PYTHON
	Py_Finalize();
	dlclose(m_pylib);
#endif
	exit_shogun();
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


int32_t CRInterface::get_int()
{
	SEXP i=get_arg_increment();

	if (i == R_NilValue || nrows(CAR(i))!=1 || ncols(CAR(i))!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	if (TYPEOF(CAR(i)) == REALSXP)
	{
		double d=REAL(CAR(i))[0];
		if (d-CMath::floor(d)!=0)
			SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);
		return (int32_t) d;
	}

	if (TYPEOF(CAR(i)) != INTSXP)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return INTEGER(CAR(i))[0];
}

float64_t CRInterface::get_real()
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


char* CRInterface::get_string(int32_t& len)
{
	SEXP s=get_arg_increment();
	if (s == R_NilValue || TYPEOF(CAR(s)) != STRSXP || length(CAR(s))!=1)
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	SEXPREC* rstr= STRING_ELT(CAR(s),0);
	const char* str= CHAR(rstr);
	len=LENGTH(rstr);
	ASSERT(len>0);
	char* res=new char[len+1];
	memcpy(res, str, len*sizeof(char));
	res[len]='\0';
	return res;
}

void CRInterface::get_byte_vector(uint8_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_char_vector(char*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_int_vector(int32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;

	SEXP rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Integer Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new int32_t[len];
	ASSERT(vec);

	for (int32_t i=0; i<len; i++)
		vec[i]= (int32_t) INTEGER(rvec)[i];
}

void CRInterface::get_shortreal_vector(float32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_real_vector(float64_t*& vec, int32_t& len)
{
	SEXP rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != REALSXP && TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new float64_t[len];
	ASSERT(vec);

	for (int32_t i=0; i<len; i++)
		vec[i]= (float64_t) REAL(rvec)[i];
}

void CRInterface::get_short_vector(int16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_word_vector(uint16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}


void CRInterface::get_byte_matrix(uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_char_matrix(char*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_int_matrix(int32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_shortreal_matrix(float32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_real_matrix(float64_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	SEXP feat=CAR(get_arg_increment());
	if( TYPEOF(feat) != REALSXP && TYPEOF(feat) != INTSXP )
		SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);

	num_vec = ncols(feat);
	num_feat = nrows(feat);
	matrix=new float64_t[num_vec*num_feat];
	ASSERT(matrix);

	for (int32_t i=0; i<num_vec; i++)
	{
		for (int32_t j=0; j<num_feat; j++)
			matrix[i*num_feat+j]= (float64_t) REAL(feat)[i*num_feat+j];
	}
}

void CRInterface::get_short_matrix(int16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_word_matrix(uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_real_sparsematrix(TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_byte_string_list(T_STRING<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_char_string_list(T_STRING<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	SEXP strs=get_arg_increment();

	if (strs == R_NilValue || TYPEOF(CAR(strs)) != STRSXP)
		SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);
	strs=CAR(strs);

	max_string_len=0;
	num_str=length(strs);
	strings=new T_STRING<char>[num_str];
	ASSERT(strings);

	for (int32_t i=0; i<num_str; i++)
	{
		SEXPREC* s= STRING_ELT(strs,i);
		char* c= (char*) CHAR(s);
		int32_t len=LENGTH(s);

		if (len && c)
		{
			char* dst=new char[len+1];
			strings[i].string=(char*) memcpy(dst, c, len*sizeof(char));
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

void CRInterface::get_int_string_list(T_STRING<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_short_string_list(T_STRING<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_word_string_list(T_STRING<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}


void CRInterface::get_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* &attrs)
{
	attrs=NULL;
}

/** set functions - to pass data from shogun to the target interface */
bool CRInterface::create_return_values(int32_t num)
{
	if (num<=0)
		return true;

	PROTECT(m_lhs=allocVector(VECSXP, num));
	m_nlhs=num;
	return length(m_lhs) == num;
}

SEXP CRInterface::get_return_values()
{
	if (m_nlhs==1)
	{
		SEXP arg=VECTOR_ELT(m_lhs, 0);
		SET_VECTOR_ELT(m_lhs, 0, R_NilValue);
		UNPROTECT(1);
		return arg;
	}

	if (m_nlhs>0)
		UNPROTECT(1);

	return m_lhs;
}


/** set functions - to pass data from shogun to the target interface */

void CRInterface::set_int(int32_t scalar)
{
	set_arg_increment(ScalarInteger(scalar));
}

void CRInterface::set_real(float64_t scalar)
{
	set_arg_increment(ScalarReal(scalar));
}

void CRInterface::set_bool(bool scalar)
{
	set_arg_increment(ScalarLogical(scalar));
}


void CRInterface::set_char_vector(const char* vec, int32_t len)
{
}

#undef SET_VECTOR
#define SET_VECTOR(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CRInterface::function_name(const sg_type* vec, int32_t len)	\
{																\
	SEXP feat=NULL;												\
	PROTECT( feat = allocVector(r_type, len) );					\
																\
	for (int32_t i=0; i<len; i++)									\
		r_cast(feat)[i]=(if_type) vec[i];						\
																\
	UNPROTECT(1);												\
	set_arg_increment(feat);									\
}

SET_VECTOR(set_byte_vector, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_VECTOR(set_int_vector, INTSXP, INTEGER, int32_t, int, "Integer")
SET_VECTOR(set_short_vector, INTSXP, INTEGER, int16_t, int, "Short")
SET_VECTOR(set_shortreal_vector, REALSXP, REAL, float32_t, float, "Single Precision")
SET_VECTOR(set_real_vector, REALSXP, REAL, float64_t, double, "Double Precision")
SET_VECTOR(set_word_vector, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_VECTOR

void CRInterface::set_char_matrix(const char* matrix, int32_t num_feat, int32_t num_vec)
{
}

#define SET_MATRIX(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CRInterface::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{																			\
	SEXP feat=NULL;															\
	PROTECT( feat = allocMatrix(r_type, num_feat, num_vec) );				\
																			\
	for (int32_t i=0; i<num_vec; i++)											\
	{																		\
		for (int32_t j=0; j<num_feat; j++)										\
			r_cast(feat)[i*num_feat+j]=(if_type) matrix[i*num_feat+j];		\
	}																		\
																			\
	UNPROTECT(1);															\
	set_arg_increment(feat);												\
}
SET_MATRIX(set_byte_matrix, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_MATRIX(set_int_matrix, INTSXP, INTEGER, int32_t, int, "Integer")
SET_MATRIX(set_short_matrix, INTSXP, INTEGER, int16_t, int, "Short")
SET_MATRIX(set_shortreal_matrix, REALSXP, REAL, float32_t, float, "Single Precision")
SET_MATRIX(set_real_matrix, REALSXP, REAL, float64_t, double, "Double Precision")
SET_MATRIX(set_word_matrix, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_MATRIX

void CRInterface::set_real_sparsematrix(const TSparse<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	// R does not support sparse matrices yet
}

void CRInterface::set_byte_string_list(const T_STRING<uint8_t>* strings, int32_t num_str)
{
}
 //this function will fail for strings containing 0, unclear how to do 'raw'
 //strings in R
void CRInterface::set_char_string_list(const T_STRING<char>* strings, int32_t num_str)
{
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	SEXP feat=NULL;
	PROTECT( feat = allocVector(STRSXP, num_str) );

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len=strings[i].length;
		if (len>0)
			SET_STRING_ELT(feat, i, mkChar(strings[i].string));
	}
	UNPROTECT(1);
	set_arg_increment(feat);
}

void CRInterface::set_int_string_list(const T_STRING<int32_t>* strings, int32_t num_str)
{
}

void CRInterface::set_short_string_list(const T_STRING<int16_t>* strings, int32_t num_str)
{
}

void CRInterface::set_word_string_list(const T_STRING<uint16_t>* strings, int32_t num_str)
{
}

void CRInterface::set_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* attrs)
{
}

bool CRInterface::cmd_run_python()
{
#ifdef HAVE_PYTHON
	return CPythonInterface::run_python_helper(this);
#else
	return false;
#endif
}

/* The main function of the shogun R interface. All commands from the R command line
 * to the shogun backend are passed using the syntax:
 * .External("sg", "func", ... ) 
 * where '...' is a number of arguments passed to the shogun function 'func'. */

extern "C" {
/* This method is called by R when the shogun module is loaded into R
 * via dyn.load('sg.so'). */

SEXP Rsg(SEXP args);

#ifdef HAVE_ELWMS
void R_init_elwms(DllInfo *info)
#else
void R_init_sg(DllInfo *info)
#endif
{
   /* There are four different external language call mechanisms available in R, namely:
    *    .C
    *    .Call
    *    .Fortran
    *    .External
    *
    * Currently shogun uses only the .External interface. */

   R_CMethodDef cMethods[] = { {NULL, NULL, 0} };
   R_FortranMethodDef fortranMethods[] = { {NULL, NULL, 0} };
   
#ifdef HAVE_ELWMS
   R_ExternalMethodDef externalMethods[] = { {"elwms", (void*(*)()) &Rsg, 1}, {NULL, NULL, 0} };
#else
   R_ExternalMethodDef externalMethods[] = { {"sg", (void*(*)()) &Rsg, 1}, {NULL, NULL, 0} };
#endif
   R_CallMethodDef callMethods[] = { {NULL, NULL, 0} };

   /* Register the routines saved in the callMethods structure so that they are available under R. */
   R_registerRoutines(info, cMethods, callMethods, (R_FortranMethodDef*) fortranMethods, (R_ExternalMethodDef*) externalMethods);

}

SEXP Rsg(SEXP args)
{
	/* The SEXP (Simple Expression) args is a list of arguments of the .External call. 
	 * it consists of "sg", "func" and additional arguments.
	 * */

	try
	{
		if (!interface)
		{
			// init_shogun has to be called before anything else
			// exit_shogun is called upon destruction of the interface (see
			// destructor of CRInterface
			init_shogun(&r_print_message, &r_print_warning,
					&r_print_error, &r_cancel_computations);
			interface=new CRInterface(args);
			ASSERT(interface);
		}
		else
			((CRInterface*) interface)->reset(args);

		if (!interface->handle())
			SG_SERROR("Unknown command.\n");
	}
	catch (std::bad_alloc)
	{
		error("Out of memory error.\n");
		return R_NilValue;
	}
	catch (ShogunException e)
	{
		error("%s", e.get_exception_string());
		return R_NilValue;
	}
	catch (...)
	{
		error("%s", "Returning from SHOGUN in error.");
		return R_NilValue;
	}

	return ((CRInterface*) interface)->get_return_values();
}

/* This method is called form within R when the current module is unregistered.
 * Note that R does not allow unregistering of single symbols. */

#ifdef HAVE_ELWMS
void R_unload_elwms(DllInfo *info)
#else
void R_unload_sg(DllInfo *info)
#endif
{
	exit_shogun();
}

} // extern "C"
