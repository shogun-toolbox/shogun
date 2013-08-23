#include "RInterface.h"

extern "C" {
#include <R.h>
#include <Rinternals.h>
#include <Rdefines.h>
#include <R_ext/Rdynload.h>
#include <Rembedded.h>
#include <Rinterface.h>
#include <R_ext/RS.h>
#include <R_ext/Error.h>
}


#include <stdlib.h>
#include <stdio.h>
#include <shogun/ui/SGInterface.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/io/SGIO.h>
#include <shogun/base/init.h>

#ifdef HAVE_PYTHON
#include "../python_static/PythonInterface.h"
#endif

#ifdef HAVE_OCTAVE
#include "../octave_static/OctaveInterface.h"
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

CRInterface::CRInterface(SEXP prhs, bool skip)
: CSGInterface(skip)
{
	skip_value=skip;
	reset(prhs);
}

CRInterface::~CRInterface()
{
}

void CRInterface::reset(SEXP prhs)
{
	CSGInterface::reset();

	if (skip_value && prhs)
		prhs=CDR(prhs);

	m_nlhs=0;
	m_nrhs=0;
	if (prhs)
		m_nrhs=Rf_length(prhs);
	if (m_nrhs<0)
		m_nrhs=0;
	m_lhs=R_NilValue;
	m_rhs=prhs;
}


/** get functions - to pass data from the target interface to shogun */


/// get type of current argument (does not increment argument counter)
IFType CRInterface::get_argument_type()
{
	if (m_rhs)
	{
		SEXP arg=CAR(m_rhs);

		switch (TYPEOF(arg))
		{
			case INTSXP:
				return DENSE_INT;
			case REALSXP:
				return DENSE_REAL;
			case STRSXP:
				return STRING_CHAR;
		};
	}
	return UNDEFINED;
}


int32_t CRInterface::get_int()
{
	SEXP i=get_arg_increment();

	if (i == R_NilValue || nrows(i)!=1 || ncols(i)!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	if (TYPEOF(i) == REALSXP)
	{
		double d=REAL(i)[0];
		if (d-CMath::floor(d)!=0)
			SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);
		return (int32_t) d;
	}

	if (TYPEOF(i) != INTSXP)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return INTEGER(i)[0];
}

float64_t CRInterface::get_real()
{
	SEXP f=get_arg_increment();
	if (f == R_NilValue || TYPEOF(f) != REALSXP || nrows(f)!=1 || ncols(f)!=1)
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return REAL(f)[0];
}

bool CRInterface::get_bool()
{
	SEXP b=get_arg_increment();
	if (b == R_NilValue || TYPEOF(b) != LGLSXP || nrows(b)!=1 || ncols(b)!=1)
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return INTEGER(b)[0] != 0;
}


char* CRInterface::get_string(int32_t& len)
{
	SEXP s=get_arg_increment();
	if (s == R_NilValue || TYPEOF(s) != STRSXP || Rf_length(s)!=1)
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	SEXPREC* rstr= STRING_ELT(s,0);
	const char* str= CHAR(rstr);
	len=LENGTH(rstr);
	ASSERT(len>0);
	char* res=SG_MALLOC(char, len+1);
	memcpy(res, str, len*sizeof(char));
	res[len]='\0';
	return res;
}

void CRInterface::get_vector(uint8_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_vector(char*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_vector(int32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;

	SEXP rvec=get_arg_increment();
	if( TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Integer Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=SG_MALLOC(int32_t, len);
	ASSERT(vec);

	for (int32_t i=0; i<len; i++)
		vec[i]= (int32_t) INTEGER(rvec)[i];
}

void CRInterface::get_vector(float32_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_vector(float64_t*& vec, int32_t& len)
{
	SEXP rvec=get_arg_increment();
	if( TYPEOF(rvec) != REALSXP && TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=SG_MALLOC(float64_t, len);
	ASSERT(vec);

	for (int32_t i=0; i<len; i++)
		vec[i]= (float64_t) REAL(rvec)[i];
}

void CRInterface::get_vector(int16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}

void CRInterface::get_vector(uint16_t*& vec, int32_t& len)
{
	vec=NULL;
	len=0;
}


void CRInterface::get_matrix(uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_matrix(char*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_matrix(int32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_matrix(float32_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_matrix(float64_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	SEXP feat=get_arg_increment();
	if( TYPEOF(feat) != REALSXP && TYPEOF(feat) != INTSXP )
		SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);

	num_vec = ncols(feat);
	num_feat = nrows(feat);
	matrix=SG_MALLOC(float64_t, num_vec*num_feat);
	ASSERT(matrix);

	for (int32_t i=0; i<num_vec; i++)
	{
		for (int32_t j=0; j<num_feat; j++)
			matrix[i*num_feat+j]= (float64_t) REAL(feat)[i*num_feat+j];
	}
}

void CRInterface::get_matrix(int16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_matrix(uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CRInterface::get_sparse_matrix(SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
}

void CRInterface::get_string_list(SGString<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_string_list(SGString<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	SEXP strs=get_arg_increment();

	if (strs == R_NilValue || TYPEOF(strs) != STRSXP)
		SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);

	SG_DEBUG("nrows=%d ncols=%d Rf_length=%d\n", nrows(strs), ncols(strs), Rf_length(strs));

	if (nrows(strs) && ncols(strs)!=1)
	{
		num_str = ncols(strs);
		max_string_len = nrows(strs);

		strings=SG_MALLOC(SGString<char>, num_str);
		ASSERT(strings);

		for (int32_t i=0; i<num_str; i++)
		{
			char* dst=SG_MALLOC(char, max_string_len+1);
			for (int32_t j=0; j<max_string_len; j++)
			{
				SEXPREC* s= STRING_ELT(strs,i*max_string_len+j);
				if (LENGTH(s)!=1)
					SG_ERROR("LENGTH(s)=%d != 1, nrows(strs)=%d ncols(strs)=%d\n", LENGTH(s), nrows(strs), ncols(strs));
				dst[j]=CHAR(s)[0];
			}
			strings[i].string=dst;
			strings[i].string[max_string_len]='\0';
			strings[i].slen=max_string_len;
		}
	}
	else
	{
		max_string_len=0;
		num_str=Rf_length(strs);
		strings=SG_MALLOC(SGString<char>, num_str);
		ASSERT(strings);

		for (int32_t i=0; i<num_str; i++)
		{
			SEXPREC* s= STRING_ELT(strs,i);
			char* c= (char*) CHAR(s);
			int32_t len=LENGTH(s);

			if (len && c)
			{
				char* dst=SG_MALLOC(char, len+1);
				strings[i].string=(char*) memcpy(dst, c, len*sizeof(char));
				strings[i].string[len]='\0';
				strings[i].slen=len;
				max_string_len=CMath::max(max_string_len, len);
			}
			else
			{
				SG_WARNING( "string with index %d has zero length\n", i+1);
				strings[i].string=0;
				strings[i].slen=0;
			}
		}
	}
}

void CRInterface::get_string_list(SGString<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_string_list(SGString<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CRInterface::get_string_list(SGString<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
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
	return Rf_length(m_lhs) == num;
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


void CRInterface::set_vector(const char* vec, int32_t len)
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

SET_VECTOR(set_vector, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_VECTOR(set_vector, INTSXP, INTEGER, int32_t, int, "Integer")
SET_VECTOR(set_vector, INTSXP, INTEGER, int16_t, int, "Short")
SET_VECTOR(set_vector, REALSXP, REAL, float32_t, float, "Single Precision")
SET_VECTOR(set_vector, REALSXP, REAL, float64_t, double, "Double Precision")
SET_VECTOR(set_vector, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_VECTOR

void CRInterface::set_matrix(const char* matrix, int32_t num_feat, int32_t num_vec)
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
SET_MATRIX(set_matrix, INTSXP, INTEGER, uint8_t, int, "Byte")
SET_MATRIX(set_matrix, INTSXP, INTEGER, int32_t, int, "Integer")
SET_MATRIX(set_matrix, INTSXP, INTEGER, int16_t, int, "Short")
SET_MATRIX(set_matrix, REALSXP, REAL, float32_t, float, "Single Precision")
SET_MATRIX(set_matrix, REALSXP, REAL, float64_t, double, "Double Precision")
SET_MATRIX(set_matrix, INTSXP, INTEGER, uint16_t, int, "Word")
#undef SET_MATRIX

void CRInterface::set_sparse_matrix(const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	// R does not support sparse matrices yet
}

void CRInterface::set_string_list(const SGString<uint8_t>* strings, int32_t num_str)
{
}
 //this function will fail for strings containing 0, unclear how to do 'raw'
 //strings in R
void CRInterface::set_string_list(const SGString<char>* strings, int32_t num_str)
{
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	SEXP feat=NULL;
	PROTECT( feat = allocVector(STRSXP, num_str) );

	for (int32_t i=0; i<num_str; i++)
	{
		int32_t len=strings[i].slen;
		if (len>0)
			SET_STRING_ELT(feat, i, mkChar(strings[i].string));
	}
	UNPROTECT(1);
	set_arg_increment(feat);
}

void CRInterface::set_string_list(const SGString<int32_t>* strings, int32_t num_str)
{
}

void CRInterface::set_string_list(const SGString<int16_t>* strings, int32_t num_str)
{
}

void CRInterface::set_string_list(const SGString<uint16_t>* strings, int32_t num_str)
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

bool CRInterface::cmd_run_octave()
{
#ifdef HAVE_OCTAVE
	return COctaveInterface::run_octave_helper(this);
#else
	return false;
#endif
}

void CRInterface::run_r_init()
{
#ifdef R_HOME_ENV
	setenv("R_HOME", R_HOME_ENV, 0);
#endif
	char* name=strdup("R");
	char* opts=strdup("-q");
	char* argv[2]={name, opts};
	Rf_initEmbeddedR(2, argv);
	free(opts);
	free(name);
}

void CRInterface::run_r_exit()
{
	//R_dot_Last();
	//R_RunExitFinalizers();
	//R_gc();
	Rf_endEmbeddedR(0);
}

bool CRInterface::run_r_helper(CSGInterface* from_if)
{
	char* rfile=NULL;

	try
	{
		for (int i=0; i<from_if->get_nrhs(); i++)
		{
			int len=0;
			char* var_name = from_if->get_string(len);
			SG_OBJ_DEBUG(from_if, "var_name = '%s'\n", var_name);
			if (strmatch(var_name, "rfile"))
			{
				len=0;
				rfile=from_if->get_string(len);
				SG_OBJ_DEBUG(from_if, "rfile = '%s'\n", rfile);
				break;
			}
			else
			{
				CRInterface* in = new CRInterface(R_NilValue, false);
				in->create_return_values(1);
				from_if->translate_arg(from_if, in);

				setVar(install(var_name), in->get_return_values(), R_GlobalEnv);
				SG_FREE(var_name);
				SG_UNREF(in);
			}
		}
	}
	catch (ShogunException e)
	{
		SG_OBJ_PRINT(from_if, "%s", e.get_exception_string())
		return true;
	}
	
	// Find source function
	SEXP src = Rf_findFun(Rf_install("source"), R_GlobalEnv);
	PROTECT(src);

	// Make file argument
	SEXP file;
	PROTECT(file = NEW_CHARACTER(1));
	SET_STRING_ELT(file, 0, COPY_TO_USER_STRING(rfile));

	// expression source(file,print.eval=p)
	SEXP expr;
	PROTECT(expr = allocVector(LANGSXP,2));
	SETCAR(expr,src); 
	SETCAR(CDR(expr),file);

	int err=0;
	R_tryEval(expr,NULL,&err);

	if (err)
	{
		UNPROTECT(3);
		SG_OBJ_PRINT(from_if, "Error occurred\n");
		return true;
	}

	SEXP results;
	PROTECT(results=findVar(install("results"), R_GlobalEnv));
	SG_OBJ_DEBUG(from_if, "Found type %d\n", TYPEOF(results));

	try
	{
		if (TYPEOF(results)==LISTSXP)
		{
			int32_t sz=Rf_length(results);
			SG_OBJ_DEBUG(from_if, "Found %d args\n", sz);

			if (sz>0 && from_if->create_return_values(sz))
			{
				CRInterface* out = new CRInterface(results, false);

				//process d
				for (int32_t i=0; i<sz; i++)
					from_if->translate_arg(out, from_if);

				SG_UNREF(out);
			}
			else if (sz!=from_if->get_nlhs())
			{
				UNPROTECT(4);
				SG_OBJ_PRINT(from_if, "Number of return values (%d) does not match "
						"number of expected return values (%d).\n",
						sz, from_if->get_nlhs());
				return true;
			}
		}
	}
	catch (ShogunException e)
	{
		UNPROTECT(4);
		SG_OBJ_PRINT(from_if, "%s", e.get_exception_string());
	}

	UNPROTECT(4);

	return true;
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
#ifdef HAVE_PYTHON
	CPythonInterface::run_python_init();
#endif
#ifdef HAVE_OCTAVE
	COctaveInterface::run_octave_init();
#endif
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
#ifdef HAVE_PYTHON
	CPythonInterface::run_python_exit();
#endif
#ifdef HAVE_OCTAVE
	COctaveInterface::run_octave_exit();
#endif

	exit_shogun();
}

} // extern "C"
