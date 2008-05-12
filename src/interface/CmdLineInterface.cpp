#include "lib/config.h"

#if defined(HAVE_CMDLINE)

#include <string>
#include <fstream>
using namespace std;

#include "interface/CmdLineInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/SimpleFile.h"


extern CSGInterface* interface;

CCmdLineInterface::CCmdLineInterface()
: CSGInterface(), m_lhs(NULL), m_rhs(NULL)
{
	reset();
}

CCmdLineInterface::~CCmdLineInterface()
{
	delete m_rhs;
}

void CCmdLineInterface::reset(const CHAR* line)
{
	CSGInterface::reset();

	if (!line)
		return;

	CHAR* pos_sep=NULL;
	CHAR delim_equal[]="=";

	// split lhs from rhs
	pos_sep=strstr(line, delim_equal);
	if (pos_sep)
	{
		//FIXME
		m_lhs=NULL;
		m_rhs=NULL;
		io.not_implemented();
	}
	else
	{
		m_lhs=NULL;

		CHAR delim_space[]=" ";
		CHAR* element=strtok((CHAR*) line, delim_space);
		if (element)
		{
			m_rhs=new CDynamicArray<CHAR*>();
			m_rhs->append_element(element);
			m_nrhs++;
			while ((element=strtok(NULL, delim_space)))
			{
				m_rhs->append_element(element);
				m_nrhs++;
			}
		}
		else
			m_rhs=NULL;
	}

/*
	for (INT i=0; i<m_rhs->get_num_elements(); i++)
		SG_PRINT("element rhs %i %s\n", i, m_rhs->get_element(i));
*/
}


/** get functions - to pass data from the target interface to shogun */


IFType CCmdLineInterface::get_argument_type()
{
	return UNDEFINED;
}


INT CCmdLineInterface::get_int()
{
	const CHAR* i=get_arg_increment();
	if (!i)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	INT value=-1;
	INT num=sscanf(i, "%d", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	return value;
}

DREAL CCmdLineInterface::get_real()
{
	const CHAR* r=get_arg_increment();
	if (!r)
		SG_ERROR("Expected Scalar Real as argument %d\n", m_rhs_counter);

	DREAL value=-1;
	INT num=sscanf(r, "%lf", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Real as argument %d\n", m_rhs_counter);

	return value;
}

bool CCmdLineInterface::get_bool()
{
	const CHAR* b=get_arg_increment();
	if (!b)
		SG_ERROR("Expected Scalar Bool as argument %d\n", m_rhs_counter);

	INT value=-1;
	INT num=sscanf(b, "%i", &value);
	if (num!=1)
		SG_ERROR("Expected Scalar Bool as argument %d\n", m_rhs_counter);

	return (value!=0);
}


CHAR* CCmdLineInterface::get_string(INT& len)
{
	const CHAR* s=get_arg_increment();
	if (!s)
		SG_ERROR("Expected 1 String as argument %d.\n", m_rhs_counter);

	len=strlen(s);
	ASSERT(len>0);

	CHAR* result=new CHAR[len+1];
	memcpy(result, s, len*sizeof(CHAR));
	result[len]='\0';
	//SG_PRINT("str %s\n", result);

	return result;
}

void CCmdLineInterface::get_byte_vector(BYTE*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_char_vector(CHAR*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_int_vector(INT*& vec, INT& len)
{
	vec=NULL;
	len=0;
/*
	vec=NULL;
	len=0;

	void* rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Integer Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new INT[len];
	ASSERT(vec);

	for (INT i=0; i<len; i++)
		vec[i]= (INT) INTEGER(rvec)[i];
		*/
}

void CCmdLineInterface::get_shortreal_vector(SHORTREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_real_vector(DREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
/*
	void* rvec=CAR(get_arg_increment());
	if( TYPEOF(rvec) != XP && TYPEOF(rvec) != INTSXP )
		SG_ERROR("Expected Double Vector as argument %d\n", m_rhs_counter);

	len=LENGTH(rvec);
	vec=new DREAL[len];
	ASSERT(vec);

	for (INT i=0; i<len; i++)
		vec[i]= (DREAL) REAL(rvec)[i];
		*/
}

void CCmdLineInterface::get_short_vector(SHORT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void CCmdLineInterface::get_word_vector(WORD*& vec, INT& len)
{
	vec=NULL;
	len=0;
}


void CCmdLineInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;

/*
	void* feat=CAR(get_arg_increment());
	if( TYPEOF(feat) != XP && TYPEOF(feat) != INTSXP )
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
	*/
}

void CCmdLineInterface::get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}


void CCmdLineInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	matrix=NULL;
	num_feat=0;
	num_vec=0;
}

void CCmdLineInterface::get_byte_string_list(T_STRING<BYTE>*& strings, INT& num_str, INT& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_char_string_list(T_STRING<CHAR>*& strings, INT& num_str, INT& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
	/*
	void* strs=get_arg_increment();

	if (strs == R_NilValue || TYPEOF(CAR(strs)) != STRSXP)
		SG_ERROR("Expected String List as argument %d\n", m_rhs_counter);
	strs=CAR(strs);

	max_string_len=0;
	num_str=length(strs);
	strings=new T_STRING<CHAR>[num_str];
	ASSERT(strings);

	for (int i=0; i<num_str; i++)
	{
		void*REC* s= STRING_ELT(strs,i);
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
	*/
}

void CCmdLineInterface::get_int_string_list(T_STRING<INT>*& strings, INT& num_str, INT& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_short_string_list(T_STRING<SHORT>*& strings, INT& num_str, INT& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CCmdLineInterface::get_word_string_list(T_STRING<WORD>*& strings, INT& num_str, INT& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

/** set functions - to pass data from shogun to the target interface */
bool CCmdLineInterface::create_return_values(INT num)
{
	if (num<=0)
		return true;

	return false;
/*
	PROTECT(m_lhs=allocVector(VECSXP, num));
	m_nlhs=num;
	return length(m_lhs) == num;
	*/
}

void* CCmdLineInterface::get_return_values()
{
	return NULL;
/*
	if (m_nlhs>0)
		UNPROTECT(1);

	if (m_nlhs==1)
	{
		void* arg=VECTOR_ELT(m_lhs, 0);
		SET_VECTOR_ELT(m_lhs, m_lhs_counter, R_NilValue);
		return arg;
	}
	return m_lhs;
	*/
}


/** set functions - to pass data from shogun to the target interface */

void CCmdLineInterface::set_int(INT scalar)
{
	//set_arg_increment(ScalarInteger(scalar));
}

void CCmdLineInterface::set_real(DREAL scalar)
{
	//set_arg_increment(ScalarReal(scalar));
}

void CCmdLineInterface::set_bool(bool scalar)
{
	//set_arg_increment(ScalarLogical(scalar));
}


void CCmdLineInterface::set_char_vector(const CHAR* vec, INT len)
{
}

void CCmdLineInterface::set_short_vector(const SHORT* vec, INT len)
{
}

void CCmdLineInterface::set_byte_vector(const BYTE* vec, INT len)
{
}

void CCmdLineInterface::set_int_vector(const INT* vec, INT len)
{
}

void CCmdLineInterface::set_shortreal_vector(const SHORTREAL* vec, INT len)
{
}

void CCmdLineInterface::set_real_vector(const DREAL* vec, INT len)
{
}

void CCmdLineInterface::set_word_vector(const WORD* vec, INT len)
{
}

/*
#undef SET_VECTOR
#define SET_VECTOR(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CCmdLineInterface::function_name(const sg_type* vec, INT len)	\
{																\
	void* feat=NULL;												\
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
SET_VECTOR(set_shortreal_vector, XP, REAL, SHORTREAL, float, "Single Precision")
SET_VECTOR(set_real_vector, XP, REAL, DREAL, double, "Double Precision")
SET_VECTOR(set_word_vector, INTSXP, INTEGER, WORD, int, "Word")
#undef SET_VECTOR
*/


void CCmdLineInterface::set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)
{
}
void CCmdLineInterface::set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)
{
}

/*
#define SET_MATRIX(function_name, r_type, r_cast, sg_type, if_type, error_string) \
void CCmdLineInterface::function_name(const sg_type* matrix, INT num_feat, INT num_vec) \
{																			\
	void* feat=NULL;															\
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
SET_MATRIX(set_shortreal_matrix, XP, REAL, SHORTREAL, float, "Single Precision")
SET_MATRIX(set_real_matrix, XP, REAL, DREAL, double, "Double Precision")
SET_MATRIX(set_word_matrix, INTSXP, INTEGER, WORD, int, "Word")
#undef SET_MATRIX
*/


void CCmdLineInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec, LONG nnz)
{
}

void CCmdLineInterface::set_byte_string_list(const T_STRING<BYTE>* strings, INT num_str)
{
}

void CCmdLineInterface::set_char_string_list(const T_STRING<CHAR>* strings, INT num_str)
{
/*
	if (!strings)
		SG_ERROR("Given strings are invalid.\n");

	void* feat=NULL;
	PROTECT( feat = allocVector(STRSXP, num_str) );

	for (INT i=0; i<num_str; i++)
	{
		INT len=strings[i].length;
		if (len>0)
			SET_STRING_ELT(feat, i, mkChar(strings[i].string));
	}
	UNPROTECT(1);
	set_arg_increment(feat);
	*/
}

void CCmdLineInterface::set_int_string_list(const T_STRING<INT>* strings, INT num_str)
{
}

void CCmdLineInterface::set_short_string_list(const T_STRING<SHORT>* strings, INT num_str)
{
}

void CCmdLineInterface::set_word_string_list(const T_STRING<WORD>* strings, INT num_str)
{
}


bool CCmdLineInterface::skip_line(const CHAR* line)
{
	if (!line)
		return true;

	if (line[0]=='\n' ||
		(line[0]=='\r' && line[1]=='\n') ||
		int(line[0])==0)
	{
		return true;
	}

	//SG_PRINT("ascii(0) %d, %c\n", int(line[0]), line[0]);

	CHAR* skipped=CIO::skip_blanks((CHAR*) line);
	if (skipped[0]=='#' || skipped[0]=='%')
		return true;

	return false;
}


int main(int argc, char* argv[])
{
	if (argc!=2)
		SG_ERROR("Need a command filename as argument.\n");

	ifstream cmdfile(argv[1]);
	if (!cmdfile.is_open())
		SG_ERROR("Could not open command file %s.\n", argv[1]);

	interface=new CCmdLineInterface();
	ASSERT(interface);

	string line;
	while (getline(cmdfile, line))
	{
		SG_PRINT("%s\n", line.c_str());
		if (((CCmdLineInterface*) interface)->skip_line(line.c_str()))
			continue;

		((CCmdLineInterface*) interface)->reset(line.c_str());
		if (!interface->handle())
			SG_ERROR("Unknown command.\n");
	}

	delete interface;
	cmdfile.close();
	return 0;
}

#endif // HAVE_CMDLINE
