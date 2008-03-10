#include "lib/config.h"

#if defined(HAVE_OCTAVE) && !defined(HAVE_SWIG)            

#include "interface/OctaveInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"

#include <octave/config.h>

#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>

extern CSGInterface* interface;

COctaveInterface::COctaveInterface(octave_value_list prhs, INT nlhs) : CSGInterface()
{
	m_nlhs=nlhs;
	m_nrhs=prhs.length();
	m_lhs=octave_value_list();
	m_rhs=prhs;
}

COctaveInterface::~COctaveInterface()
{
}

/** get functions - to pass data from the target interface to shogun */
void COctaveInterface::parse_args(INT num_args, INT num_default_args)
{
}


/// get type of current argument (does not increment argument counter)
IFType COctaveInterface::get_argument_type()
{
	return UNDEFINED;
}


INT COctaveInterface::get_int()
{
	const octave_value i=get_arg_increment();
	if (!i.is_real_scalar())
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	double s=i.double_value();
	if (s-CMath::floor(s)!=0)
		SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);

	return INT(s);
}

DREAL COctaveInterface::get_real()
{
	const octave_value f=get_arg_increment();
	if (!f.is_real_scalar())
		SG_ERROR("Expected Scalar Float as argument %d\n", m_rhs_counter);

	return f.double_value();
}

bool COctaveInterface::get_bool()
{
	const octave_value b=get_arg_increment();
	if (b.is_scalar_type())
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return b.bool_value();
}

CHAR* COctaveInterface::get_string(INT& len)
{
	const octave_value s=get_arg_increment();
	if (!s.is_string())
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	std::string std_str=s.string_value();
	const CHAR* str= std_str.c_str();
	len=std_str.length();
	ASSERT(str && len>0);

	CHAR* cstr = new CHAR[len+1];
	ASSERT(cstr);

	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	return cstr;
}

void COctaveInterface::get_byte_vector(BYTE*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_char_vector(CHAR*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_int_vector(INT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_shortreal_vector(SHORTREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_real_vector(DREAL*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_short_vector(SHORT*& vec, INT& len)
{
	vec=NULL;
	len=0;
}

void COctaveInterface::get_word_vector(WORD*& vec, INT& len)
{
	vec=NULL;
	len=0;
}


void COctaveInterface::get_byte_matrix(BYTE*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_char_matrix(CHAR*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_int_matrix(INT*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_shortreal_matrix(SHORTREAL*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_real_matrix(DREAL*& matrix, INT& num_feat, INT& num_vec)
{
	const octave_value mat_feat=get_arg_increment();
	if (!mat_feat.is_real_matrix())
		SG_ERROR("Expected Double Matrix as argument %d\n", m_rhs_counter);

	Matrix m = mat_feat.matrix_value();
	num_vec = m.cols();
	num_feat = m.rows();
	matrix=new DREAL[num_vec*num_feat];
	ASSERT(matrix);

	for (INT i=0; i<num_vec; i++)
		for (INT j=0; j<num_feat; j++)
			matrix[i*num_feat+j]= (double) m(j,i);
}

void COctaveInterface::get_short_matrix(SHORT*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_word_matrix(WORD*& matrix, INT& num_feat, INT& num_vec)
{
}


void COctaveInterface::get_byte_sparsematrix(TSparse<BYTE>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_char_sparsematrix(TSparse<CHAR>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_int_sparsematrix(TSparse<INT>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_shortreal_sparsematrix(TSparse<SHORTREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_short_sparsematrix(TSparse<SHORT>*& matrix, INT& num_feat, INT& num_vec)
{
}

void COctaveInterface::get_word_sparsematrix(TSparse<WORD>*& matrix, INT& num_feat, INT& num_vec)
{
}


void COctaveInterface::get_string_list(T_STRING<CHAR>*& strings, INT& num_str)
{
}


/** set functions - to pass data from shogun to the target interface */
void COctaveInterface::create_return_values(INT num_val)
{
}

void COctaveInterface::set_byte_vector(const BYTE* vec, INT len)
{
}

void COctaveInterface::set_char_vector(const CHAR* vec, INT len)
{
}

void COctaveInterface::set_int_vector(const INT* vec, INT len)
{
}

void COctaveInterface::set_shortreal_vector(const SHORTREAL* vec, INT len)
{
}

void COctaveInterface::set_real_vector(const DREAL* vec, INT len)
{
}

void COctaveInterface::set_short_vector(const SHORT* vec, INT len)
{
}

void COctaveInterface::set_word_vector(const WORD* vec, INT len)
{
}


void COctaveInterface::set_byte_matrix(const BYTE* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_char_matrix(const CHAR* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_int_matrix(const INT* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_shortreal_matrix(const SHORTREAL* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_real_matrix(const DREAL* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_short_matrix(const SHORT* matrix, INT num_feat, INT num_vec)
{
}
void COctaveInterface::set_word_matrix(const WORD* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_byte_sparsematrix(const TSparse<BYTE>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_char_sparsematrix(const TSparse<CHAR>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_int_sparsematrix(const TSparse<INT>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_shortreal_sparsematrix(const TSparse<SHORTREAL>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_short_sparsematrix(const TSparse<SHORT>* matrix, INT num_feat, INT num_vec)
{
}

void COctaveInterface::set_word_sparsematrix(const TSparse<WORD>* matrix, INT num_feat, INT num_vec)
{
}


void COctaveInterface::set_string_list(const T_STRING<CHAR>* strings, INT num_str)
{
}

void COctaveInterface::set_string_list(const T_STRING<WORD>* strings, INT num_str)
{
}

void COctaveInterface::submit_return_values()
{
}

DEFUN_DLD (sg, prhs, nlhs, "shogun.")
{
	delete interface;
	interface=new COctaveInterface(prhs, nlhs);

	try
	{
		if (!interface->handle())
			SG_ERROR("interface currently does not handle this command\n");
	}
	catch (ShogunException e)
	{
		return octave_value_list();
	}

	return ((COctaveInterface*) interface)->get_return_values();
}
#endif // HAVE_OCTAVE && !HAVE_SWIG
