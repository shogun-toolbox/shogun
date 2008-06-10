#include "lib/config.h"

#if defined(HAVE_OCTAVE) && !defined(HAVE_SWIG)

#include "interface/OctaveInterface.h"
#include "interface/SGInterface.h"

#include "lib/ShogunException.h"
#include "lib/io.h"
#include "lib/octave.h"
#include "lib/memory.h"

extern CSGInterface* interface;

COctaveInterface::COctaveInterface(octave_value_list prhs, INT nlhs)
: CSGInterface()
{
	reset(prhs, nlhs);
}

COctaveInterface::~COctaveInterface()
{
}

void COctaveInterface::reset(octave_value_list prhs, INT nlhs)
{
	CSGInterface::reset();

	m_nlhs=nlhs;
	m_nrhs=prhs.length();
	m_lhs=octave_value_list();
	m_rhs=prhs;
}

/** get functions - to pass data from the target interface to shogun */


/// get type of current argument (does not increment argument counter)
IFType COctaveInterface::get_argument_type()
{
	octave_value arg=m_rhs(m_rhs_counter);

	if (arg.is_char_matrix())
		return STRING_CHAR;
	else if (arg.is_uint8_type() && arg.is_matrix_type())
		return STRING_BYTE;

	if (arg.is_sparse_type())
	{
		if (arg.is_uint8_type())
			return SPARSE_BYTE;
		else if (arg.is_char_matrix())
			return SPARSE_CHAR;
		else if (arg.is_int32_type())
			return SPARSE_INT;
		else if (arg.is_double_type())
			return SPARSE_REAL;
		else if (arg.is_int16_type())
			return SPARSE_SHORT;
		else if (arg.is_single_type())
			return SPARSE_SHORTREAL;
		else if (arg.is_uint16_type())
			return SPARSE_WORD;
		else
			return UNDEFINED;
	}
	else if (arg.is_matrix_type())
	{
		if (arg.is_uint32_type())
			return DENSE_INT;
		else if (arg.is_double_type())
			return DENSE_REAL;
		else if (arg.is_int16_type())
			return DENSE_SHORT;
		else if (arg.is_single_type())
			return DENSE_SHORTREAL;
		else if (arg.is_uint16_type())
			return DENSE_WORD;
	}
	else if (arg.is_cell())
	{
		Cell c = arg.cell_value();

		if (c.nelem()>0)
		{
			if (c.elem(0).is_char_matrix() && c.elem(0).rows()==1)
				return STRING_CHAR;
			else if (c.elem(0).is_uint8_type() && c.elem(0).rows()==1)
				return STRING_BYTE;
			else if (c.elem(0).is_int32_type() && c.elem(0).rows()==1)
				return STRING_INT;
			else if (c.elem(0).is_int16_type() && c.elem(0).rows()==1)
				return STRING_SHORT;
			else if (c.elem(0).is_uint16_type() && c.elem(0).rows()==1)
				return STRING_WORD;
		}
	}

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
	if (b.is_bool_scalar())
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

	CHAR* cstr=new CHAR[len+1];
	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	return cstr;
}

#define GET_VECTOR(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& vec, INT& len)						\
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check() || mat_feat.rows()!=1)	\
		SG_ERROR("Expected " error_string " Vector as argument %d\n", m_rhs_counter); \
																					\
	oct_type m = mat_feat.oct_converter();											\
	len = m.cols();																	\
	vec=new sg_type[len];															\
																					\
	for (INT i=0; i<len; i++)														\
			vec[i]= (sg_type) m(i);													\
}
GET_VECTOR(get_byte_vector, is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
GET_VECTOR(get_char_vector, is_char_matrix, charMatrix, char_matrix_value, CHAR, CHAR, "Char")
GET_VECTOR(get_int_vector, is_int32_type, int32NDArray, uint8_array_value, INT, INT, "Integer")
GET_VECTOR(get_short_vector, is_int16_type, int16NDArray, uint8_array_value, SHORT, SHORT, "Short")
GET_VECTOR(get_shortreal_vector, is_single_type, Matrix, matrix_value, SHORTREAL, SHORTREAL, "Single Precision")
GET_VECTOR(get_real_vector, is_double_type, Matrix, matrix_value, DREAL, DREAL, "Double Precision")
GET_VECTOR(get_word_vector, is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef GET_VECTOR


#define GET_MATRIX(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& matrix, INT& num_feat, INT& num_vec) \
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())						\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n", m_rhs_counter); \
																					\
	oct_type m = mat_feat.oct_converter();												\
	num_vec = m.cols();																\
	num_feat = m.rows();															\
	matrix=new sg_type[num_vec*num_feat];											\
																					\
	for (INT i=0; i<num_vec; i++)													\
		for (INT j=0; j<num_feat; j++)												\
			matrix[i*num_feat+j]= (sg_type) m(j,i);									\
}
GET_MATRIX(get_byte_matrix, is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
GET_MATRIX(get_char_matrix, is_char_matrix, charMatrix, char_matrix_value, CHAR, CHAR, "Char")
GET_MATRIX(get_int_matrix, is_int32_type, int32NDArray, uint8_array_value, INT, INT, "Integer")
GET_MATRIX(get_short_matrix, is_int16_type, int16NDArray, uint8_array_value, SHORT, SHORT, "Short")
GET_MATRIX(get_shortreal_matrix, is_single_type, Matrix, matrix_value, SHORTREAL, SHORTREAL, "Single Precision")
GET_MATRIX(get_real_matrix, is_double_type, Matrix, matrix_value, DREAL, DREAL, "Double Precision")
GET_MATRIX(get_word_matrix, is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef GET_MATRIX

#define GET_NDARRAY(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& array, INT*& dims, INT& num_dims)	\
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())					\
		SG_ERROR("Expected " error_string " ND Array as argument %d\n", m_rhs_counter); \
																					\
	num_dims = (INT) mat_feat.ndims();												\
	oct_type m = mat_feat.oct_converter();											\
	LONG total_size=mat_feat.length();												\
	array=new sg_type[total_size];													\
																					\
	for (LONG i=0; i<total_size; i++)												\
		array[i]= (sg_type) m(i);													\
}
GET_NDARRAY(get_byte_ndarray, is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
GET_NDARRAY(get_char_ndarray, is_char_matrix, charMatrix, char_matrix_value, CHAR, CHAR, "Char")
GET_NDARRAY(get_int_ndarray, is_int32_type, int32NDArray, uint8_array_value, INT, INT, "Integer")
GET_NDARRAY(get_short_ndarray, is_int16_type, int16NDArray, uint8_array_value, SHORT, SHORT, "Short")
GET_NDARRAY(get_shortreal_ndarray, is_single_type, Matrix, matrix_value, SHORTREAL, SHORTREAL, "Single Precision")
GET_NDARRAY(get_real_ndarray, is_double_type, Matrix, matrix_value, DREAL, DREAL, "Double Precision")
GET_NDARRAY(get_word_ndarray, is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef GET_NDARRAY

void COctaveInterface::get_real_sparsematrix(TSparse<DREAL>*& matrix, INT& num_feat, INT& num_vec)
{
	const octave_value mat_feat=get_arg_increment();
	if (!mat_feat.is_sparse_type() || !(mat_feat.is_double_type()))
		SG_ERROR("Expected Sparse Double Matrix as argument %d\n", m_rhs_counter);

	SparseMatrix sm = mat_feat.sparse_matrix_value ();
	num_vec=sm.cols();
	num_feat=sm.rows();
	LONG nnz=sm.nelem();

	matrix=new TSparse<DREAL>[num_vec];

	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=sm.cidx(i+1)-sm.cidx(i);
		matrix[i].vec_index=i;
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=new TSparseEntry<DREAL>[len];

			for (INT j=0; j<len; j++)
			{
				matrix[i].features[j].entry=sm.data(offset);
				matrix[i].features[j].feat_index=sm.ridx(offset);
				offset++;
			}
		}
		else
			matrix[i].features=NULL;
	}
	ASSERT(offset=nnz);
}

#define GET_STRINGLIST(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(T_STRING<sg_type>*& strings, INT& num_str, INT& max_string_len) \
{																					\
	max_string_len=0;														\
	octave_value arg=get_arg_increment();											\
	if (arg.is_cell())																\
	{																				\
		Cell c = arg.cell_value();													\
		num_str=c.nelem();															\
		ASSERT(num_str>=1);															\
		strings=new T_STRING<sg_type>[num_str];										\
																					\
		for (int i=0; i<num_str; i++)												\
		{																			\
			if (!c.elem(i).oct_type_check() || !c.elem(i).rows()==1)				\
				SG_ERROR("Expected String of type " error_string " as argument %d.\n", m_rhs_counter); \
			oct_type str=c.elem(i).oct_converter();							\
																					\
			INT len=str.cols();														\
			if (len>0) 																\
			{ 																		\
				strings[i].length=len; /* all must have same length in octave */ 	\
				strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */ \
				INT j; 																\
				for (j=0; j<len; j++) 												\
					strings[i].string[j]=str(0,j); 									\
				strings[i].string[j]='\0'; 											\
				max_string_len=CMath::max(max_string_len, len);						\
			}																		\
			else																	\
			{																		\
				SG_WARNING( "string with index %d has zero length.\n", i+1);		\
				strings[i].length=0;												\
				strings[i].string=NULL;												\
			}																		\
		} 																			\
	} 																				\
	else if (arg.oct_type_check())							\
	{																				\
		oct_type data=arg.oct_converter();											\
		num_str=data.cols(); 														\
		INT len=data.rows(); 														\
		strings=new T_STRING<sg_type>[num_str]; 									\
																					\
		for (INT i=0; i<num_str; i++) 												\
		{ 																			\
			if (len>0) 																\
			{ 																		\
				strings[i].length=len; /* all must have same length in octave */ 	\
				strings[i].string=new sg_type[len+1]; /* not zero terminated in octave */ \
				INT j; 																\
				for (j=0; j<len; j++) 												\
					strings[i].string[j]=data(j,i);									\
				strings[i].string[j]='\0'; 											\
			} 																		\
			else 																	\
			{ 																		\
				SG_WARNING( "string with index %d has zero length.\n", i+1); 		\
				strings[i].length=0; 												\
				strings[i].string=NULL; 											\
			} 																		\
		} 																			\
		max_string_len=len;															\
	}																				\
	else																			\
	{\
	SG_PRINT("matrix_type: %d\n", arg.is_matrix_type() ? 1 : 0); \
		SG_ERROR("Expected String, got class %s as argument %d.\n",					\
			"???", m_rhs_counter);													\
	}\
}
GET_STRINGLIST(get_byte_string_list, is_matrix_type() && arg.is_uint8_type, uint8NDArray, uint8_array_value, BYTE, BYTE, "Byte")
GET_STRINGLIST(get_char_string_list, is_char_matrix, charMatrix, char_matrix_value, CHAR, CHAR, "Char")
GET_STRINGLIST(get_int_string_list, is_matrix_type() && arg.is_int32_type, int32NDArray, int32_array_value, INT, INT, "Integer")
GET_STRINGLIST(get_short_string_list, is_matrix_type() && arg.is_int16_type, int16NDArray, int16_array_value, SHORT, SHORT, "Short")
GET_STRINGLIST(get_word_string_list, is_matrix_type() && arg.is_uint16_type, uint16NDArray, uint16_array_value, WORD, WORD, "Word")
#undef GET_STRINGLIST


/** set functions - to pass data from shogun to Octave */

void COctaveInterface::set_int(INT scalar)
{
	octave_value o(scalar);
	set_arg_increment(o);
}

void COctaveInterface::set_real(DREAL scalar)
{
	octave_value o(scalar);
	set_arg_increment(o);
}

void COctaveInterface::set_bool(bool scalar)
{
	octave_value o(scalar);
	set_arg_increment(o);
}


#define SET_VECTOR(function_name, oct_type, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(const sg_type* vec, INT len)				\
{																				\
	oct_type mat=oct_type(dim_vector(1, len));									\
																				\
	for (INT i=0; i<len; i++)													\
			mat(i) = (if_type) vec[i];											\
																				\
	set_arg_increment(mat);														\
}
SET_VECTOR(set_byte_vector, uint8NDArray, BYTE, BYTE, "Byte")
SET_VECTOR(set_char_vector, charMatrix, CHAR, CHAR, "Char")
SET_VECTOR(set_int_vector, int32NDArray, INT, INT, "Integer")
SET_VECTOR(set_short_vector, int16NDArray, SHORT, SHORT, "Short")
SET_VECTOR(set_shortreal_vector, Matrix, SHORTREAL, SHORTREAL, "Single Precision")
SET_VECTOR(set_real_vector, Matrix, DREAL, DREAL, "Double Precision")
SET_VECTOR(set_word_vector, uint16NDArray, WORD, WORD, "Word")
#undef SET_VECTOR

#define SET_MATRIX(function_name, oct_type, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(const sg_type* matrix, INT num_feat, INT num_vec) \
{																				\
	oct_type mat=oct_type(dim_vector(num_feat, num_vec));						\
																				\
	for (INT i=0; i<num_vec; i++)												\
	{																			\
		for (INT j=0; j<num_feat; j++)											\
			mat(j,i) = (if_type) matrix[j+i*num_feat];							\
	}																			\
																				\
	set_arg_increment(mat);														\
}
SET_MATRIX(set_byte_matrix, uint8NDArray, BYTE, BYTE, "Byte")
SET_MATRIX(set_char_matrix, charMatrix, CHAR, CHAR, "Char")
SET_MATRIX(set_int_matrix, int32NDArray, INT, INT, "Integer")
SET_MATRIX(set_short_matrix, int16NDArray, SHORT, SHORT, "Short")
SET_MATRIX(set_shortreal_matrix, Matrix, SHORTREAL, SHORTREAL, "Single Precision")
SET_MATRIX(set_real_matrix, Matrix, DREAL, DREAL, "Double Precision")
SET_MATRIX(set_word_matrix, uint16NDArray, WORD, WORD, "Word")
#undef SET_MATRIX

void COctaveInterface::set_real_sparsematrix(const TSparse<DREAL>* matrix, INT num_feat, INT num_vec, LONG nnz)
{
	SparseMatrix sm((octave_idx_type) num_feat, (octave_idx_type) num_vec, (octave_idx_type) nnz);

	LONG offset=0;
	for (INT i=0; i<num_vec; i++)
	{
		INT len=matrix[i].num_feat_entries;
		sm.cidx(i)=offset;
		for (INT j=0; j<len; j++)
		{
			sm.data(offset) = matrix[i].features[j].entry;
			sm.ridx(offset) = matrix[i].features[j].feat_index;
			offset++;
		}
	}
	sm.cidx(num_vec) = offset;

	set_arg_increment(sm);
}

#define SET_STRINGLIST(function_name, oct_type, sg_type, if_type, error_string)	\
void COctaveInterface::function_name(const T_STRING<sg_type>* strings, INT num_str)	\
{																					\
	if (!strings)																	\
		SG_ERROR("Given strings are invalid.\n");									\
																					\
	Cell c= Cell(dim_vector(num_str));															\
	if (c.nelem()!=num_str)															\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str);			\
																					\
	for (INT i=0; i<num_str; i++)													\
	{																				\
		INT len=strings[i].length;													\
		if (len>0)																	\
		{																			\
			oct_type str(dim_vector(1,len));										\
			if (str.cols()!=len)													\
				SG_ERROR("Couldn't create " error_string " String %d of length %d.\n", i, len);	\
																					\
			for (INT j=0; j<len; j++)												\
				str(j)= (if_type) strings[i].string[j];								\
			c.elem(i)=str;															\
		}																			\
	}																				\
																					\
	set_arg_increment(c);															\
}
SET_STRINGLIST(set_byte_string_list, int8NDArray, BYTE, BYTE, "Byte")
SET_STRINGLIST(set_char_string_list, charNDArray, CHAR, CHAR, "Char")
SET_STRINGLIST(set_int_string_list, int32NDArray, INT, INT, "Integer")
SET_STRINGLIST(set_short_string_list, int16NDArray, SHORT, SHORT, "Short")
SET_STRINGLIST(set_word_string_list, uint16NDArray, WORD, WORD, "Word")
#undef SET_STRINGLIST

DEFUN_DLD (sg, prhs, nlhs, "shogun.")
{
	try
	{
		if (!interface)
			interface=new COctaveInterface(prhs, nlhs);
		else
			((COctaveInterface*) interface)->reset(prhs, nlhs);

		if (!interface->handle())
			SG_ERROR("Unknown command.\n");

		return ((COctaveInterface*) interface)->get_return_values();
	}
	catch (std::bad_alloc)
	{
		SG_PRINT("Out of memory error.\n");
		return octave_value_list();
	}
	catch (ShogunException e)
	{
		return octave_value_list();
	}
}
#endif // HAVE_OCTAVE && !HAVE_SWIG
