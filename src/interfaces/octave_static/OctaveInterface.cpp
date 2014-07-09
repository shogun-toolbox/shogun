#include "OctaveInterface.h"

#undef length

#include <octave/ov.h>
#include <octave/octave.h>
#include <octave/variables.h>
#include <octave/unwind-prot.h>
#include <octave/sighandlers.h>
#include <octave/sysdep.h>
#include <octave/parse.h>
#include <octave/toplev.h>
#include <octave/dim-vector.h>
#include <octave/defun-dld.h>
#include <octave/error.h>
#include <octave/oct-obj.h>
#include <octave/pager.h>
#include <octave/symtab.h>
#include <octave/variables.h>
#include <octave/Cell.h>
#include <stdio.h>

#include <shogun/ui/SGInterface.h>
#include <shogun/lib/ShogunException.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/config.h>
#include <shogun/lib/memory.h>
#include <shogun/base/init.h>

#ifdef HAVE_PYTHON
#include "../python_static/PythonInterface.h"
#endif

#ifdef HAVE_R
#include "../r_static/RInterface.h"
#undef length
#endif

void octave_print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void octave_print_warning(FILE* target, const char* str)
{
	if (target==stdout)
		::warning(str);
	else
		fprintf(target, "%s", str);
}

void octave_print_error(FILE* target, const char* str)
{
	if (target!=stdout)
		fprintf(target, "%s", str);
}

void octave_cancel_computations(bool &delayed, bool &immediately)
{
}

extern CSGInterface* interface;

COctaveInterface::COctaveInterface(octave_value_list prhs, int32_t nlhs, bool verbose)
: CSGInterface(verbose)
{
	reset(prhs, nlhs);
}

COctaveInterface::~COctaveInterface()
{
}

void COctaveInterface::reset(octave_value_list prhs, int32_t nlhs)
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

	if (arg.is_real_scalar())
		return SCALAR_REAL;
	if (arg.is_bool_scalar())
		return SCALAR_BOOL;

	if (arg.is_char_matrix())
		return STRING_CHAR;
	if (arg.is_uint8_type() && arg.is_matrix_type())
		return STRING_BYTE;

	if (arg.is_sparse_type())
	{
		if (arg.is_uint8_type())
			return SPARSE_BYTE;
		if (arg.is_char_matrix())
			return SPARSE_CHAR;
		if (arg.is_int32_type())
			return SPARSE_INT;
		if (arg.is_double_type())
			return SPARSE_REAL;
		if (arg.is_int16_type())
			return SPARSE_SHORT;
		if (arg.is_single_type())
			return SPARSE_SHORTREAL;
		if (arg.is_uint16_type())
			return SPARSE_WORD;

		return UNDEFINED;
	}

	if (arg.is_cell())
	{
		Cell c = arg.cell_value();

		if (c.nelem()>0)
		{
			if (c.elem(0).is_char_matrix() && c.elem(0).rows()==1)
				return STRING_CHAR;
			if (c.elem(0).is_uint8_type() && c.elem(0).rows()==1)
				return STRING_BYTE;
			if (c.elem(0).is_int32_type() && c.elem(0).rows()==1)
				return STRING_INT;
			if (c.elem(0).is_int16_type() && c.elem(0).rows()==1)
				return STRING_SHORT;
			if (c.elem(0).is_uint16_type() && c.elem(0).rows()==1)
				return STRING_WORD;
		}
	}


	if (arg.is_matrix_type() && arg.ndims()==1 && arg.rows()==1)
	{
		if (arg.is_uint32_type())
			return VECTOR_INT;
		if (arg.is_double_type())
			return VECTOR_REAL;
		if (arg.is_int16_type())
			return VECTOR_SHORT;
		if (arg.is_single_type())
			return VECTOR_SHORTREAL;
		if (arg.is_uint16_type())
			return VECTOR_WORD;

		return UNDEFINED;
	}

	if (arg.is_matrix_type() && arg.ndims()==2)
	{
		if (arg.is_uint32_type())
			return DENSE_INT;
		if (arg.is_double_type())
			return DENSE_REAL;
		if (arg.is_int16_type())
			return DENSE_SHORT;
		if (arg.is_single_type())
			return DENSE_SHORTREAL;
		if (arg.is_uint16_type())
			return DENSE_WORD;

		return UNDEFINED;
	}

	if (arg.is_matrix_type() && arg.ndims()>2)
	{
		if (arg.is_uint8_type())
			return NDARRAY_BYTE;
		if (arg.is_uint32_type())
			return NDARRAY_INT;
		if (arg.is_double_type())
			return NDARRAY_REAL;
		if (arg.is_int16_type())
			return NDARRAY_SHORT;
		if (arg.is_single_type())
			return NDARRAY_SHORTREAL;
		if (arg.is_uint16_type())
			return NDARRAY_WORD;

		return UNDEFINED;
	}

	if (arg.is_map())
		return ATTR_STRUCT;

	return UNDEFINED;
}


int32_t COctaveInterface::get_int()
{
	const octave_value i=get_arg_increment();
	if (!i.is_real_scalar())
		SG_ERROR("Expected Scalar Integer as argument %d\n", m_rhs_counter);

	double s=i.double_value();
	if (s-CMath::floor(s)!=0)
		SG_ERROR("Expected Integer as argument %d\n", m_rhs_counter);

	return int32_t(s);
}

float64_t COctaveInterface::get_real()
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
		return b.bool_value();
	else if (b.is_real_scalar())
		return (b.double_value()!=0);
	else
		SG_ERROR("Expected Scalar Boolean as argument %d\n", m_rhs_counter);

	return false;
}

char* COctaveInterface::get_string(int32_t& len)
{
	const octave_value s=get_arg_increment();
	if (!s.is_string())
		SG_ERROR("Expected String as argument %d\n", m_rhs_counter);

	std::string std_str=s.string_value();
	const char* str= std_str.c_str();
	len=std_str.length();
	ASSERT(str && len>0);

	char* cstr=SG_MALLOC(char, len+1);
	memcpy(cstr, str, len+1);
	cstr[len]='\0';

	return cstr;
}

#define GET_VECTOR(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& vec, int32_t& len)						\
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.oct_type_check())													\
		SG_ERROR("Expected " error_string " Vector as argument %d\n", m_rhs_counter); \
																					\
	oct_type m = mat_feat.oct_converter();											\
																					\
	if (m.rows()!=1)																\
		SG_ERROR("Expected " error_string " (1xN) Vector as argument %d, got vector " \
			"of shape (%dx%d)\n", m_rhs_counter, m.rows(), m.cols());				\
																					\
	len = m.cols();																	\
	vec=SG_MALLOC(sg_type, len);															\
																					\
	for (int32_t i=0; i<len; i++)														\
			vec[i]= (sg_type) m(i);													\
}
GET_VECTOR(get_vector, is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
GET_VECTOR(get_vector, is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
GET_VECTOR(get_vector, is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
GET_VECTOR(get_vector, is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
GET_VECTOR(get_vector, is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
GET_VECTOR(get_vector, is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
GET_VECTOR(get_vector, is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef GET_VECTOR


#define GET_MATRIX(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec) \
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.oct_type_check())						\
		SG_ERROR("Expected " error_string " Matrix as argument %d\n", m_rhs_counter); \
																					\
	oct_type m = mat_feat.oct_converter();												\
	num_vec = m.cols();																\
	num_feat = m.rows();															\
	matrix = SG_MALLOC(sg_type, num_vec*num_feat);											\
																					\
	for (int32_t i=0; i<num_vec; i++)													\
		for (int32_t j=0; j<num_feat; j++)												\
			matrix[i*num_feat+j]= (sg_type) m(j,i);									\
}
GET_MATRIX(get_matrix, is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
GET_MATRIX(get_matrix, is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
GET_MATRIX(get_matrix, is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
GET_MATRIX(get_matrix, is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
GET_MATRIX(get_matrix, is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
GET_MATRIX(get_matrix, is_double_type, Matrix, matrix_value, float64_t, float64_t, "Double Precision")
GET_MATRIX(get_matrix, is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef GET_MATRIX

#define GET_NDARRAY(function_name, oct_type_check, oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(sg_type*& array, int32_t*& dims, int32_t& num_dims)	\
{																					\
	const octave_value mat_feat=get_arg_increment();								\
	if (!mat_feat.is_matrix_type() || !mat_feat.oct_type_check())					\
		SG_ERROR("Expected " error_string " ND Array as argument %d\n", m_rhs_counter); \
																					\
	num_dims = (int32_t) mat_feat.ndims();											\
	dim_vector dimvec = mat_feat.dims();											\
																					\
	dims=SG_MALLOC(int32_t, num_dims);														\
	for (int32_t d=0; d<num_dims; d++)												\
		dims[d]=(int32_t) dimvec(d);												\
																					\
	oct_type m = mat_feat.oct_converter();											\
	int64_t total_size=m.nelem();													\
																					\
	array=SG_MALLOC(sg_type, total_size);													\
	for (int64_t i=0; i<total_size; i++)											\
		array[i]= (sg_type) m(i);													\
}
GET_NDARRAY(get_ndarray, is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
GET_NDARRAY(get_ndarray, is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
GET_NDARRAY(get_ndarray, is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
GET_NDARRAY(get_ndarray, is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
GET_NDARRAY(get_ndarray, is_single_type, Matrix, matrix_value, float32_t, float32_t, "Single Precision")
GET_NDARRAY(get_ndarray, is_double_type, NDArray, array_value, float64_t, float64_t, "Double Precision")
GET_NDARRAY(get_ndarray, is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef GET_NDARRAY

void COctaveInterface::get_sparse_matrix(SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)
{
	const octave_value mat_feat=get_arg_increment();
	if (!mat_feat.is_sparse_type() || !(mat_feat.is_double_type()))
		SG_ERROR("Expected Sparse Double Matrix as argument %d\n", m_rhs_counter);

	SparseMatrix sm = mat_feat.sparse_matrix_value ();
	num_vec=sm.cols();
	num_feat=sm.rows();
	int64_t nnz=sm.nelem();

	matrix=SG_MALLOC(SGSparseVector<float64_t>, num_vec);

	int64_t offset=0;
	for (int32_t i=0; i<num_vec; i++)
	{
		int32_t len=sm.cidx(i+1)-sm.cidx(i);
		matrix[i].num_feat_entries=len;

		if (len>0)
		{
			matrix[i].features=SG_MALLOC(SGSparseVectorEntry<float64_t>, len);

			for (int32_t j=0; j<len; j++)
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

#define GET_STRINGLIST(function_name, oct_type_check1, oct_type_check2, \
		oct_type, oct_converter, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len) \
{																					\
	max_string_len=0;																\
	octave_value arg=get_arg_increment();											\
	if (arg.is_cell())																\
	{																				\
		Cell c = arg.cell_value();													\
		num_str=c.nelem();															\
		ASSERT(num_str>=1);															\
		strings=SG_MALLOC(SGString<sg_type>, num_str);								\
																					\
		for (int32_t i=0; i<num_str; i++)											\
		{																			\
			if (!c.elem(i).oct_type_check1() || !c.elem(i).oct_type_check2()		\
					|| !c.elem(i).rows()==1)										\
				SG_ERROR("Expected String of type " error_string " as argument %d.\n", m_rhs_counter); \
																					\
			oct_type str=c.elem(i).oct_converter();									\
																					\
			int32_t len=str.cols();													\
			if (len>0)																\
			{																		\
				strings[i].slen=len; /* all must have same length in octave */		\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated in octave */ \
				int32_t j;															\
				for (j=0; j<len; j++)												\
					strings[i].string[j]=str(0,j);									\
				strings[i].string[j]='\0';											\
				max_string_len=CMath::max(max_string_len, len);						\
			}																		\
			else																	\
			{																		\
				SG_WARNING( "string with index %d has zero length.\n", i+1);		\
				strings[i].slen=0;													\
				strings[i].string=NULL;												\
			}																		\
		}																			\
	}																				\
	else if (arg.oct_type_check1() && arg.oct_type_check2())						\
	{																				\
		oct_type data=arg.oct_converter();											\
		num_str=data.cols();														\
		int32_t len=data.rows();													\
		strings=SG_MALLOC(SGString<sg_type>, num_str);								\
																					\
		for (int32_t i=0; i<num_str; i++)											\
		{																			\
			if (len>0)																\
			{																		\
				strings[i].slen=len; /* all must have same length in octave */		\
				strings[i].string=SG_MALLOC(sg_type, len+1); /* not zero terminated in octave */ \
				int32_t j;															\
				for (j=0; j<len; j++)												\
					strings[i].string[j]=data(j,i);									\
				strings[i].string[j]='\0';											\
			}																		\
			else																	\
			{																		\
				SG_WARNING( "string with index %d has zero length.\n", i+1);		\
				strings[i].slen=0;													\
				strings[i].string=NULL;											\
			}																		\
		}																			\
		max_string_len=len;															\
	}																				\
	else																			\
	{\
	SG_PRINT("matrix_type: %d\n", arg.is_matrix_type() ? 1 : 0); \
		SG_ERROR("Expected String, got class %s as argument %d.\n",					\
			"???", m_rhs_counter);													\
	}\
}
/* ignore the g++ warning here */
GET_STRINGLIST(get_string_list, is_matrix_type, is_uint8_type, uint8NDArray, uint8_array_value, uint8_t, uint8_t, "Byte")
GET_STRINGLIST(get_string_list, is_char_matrix, is_char_matrix, charMatrix, char_matrix_value, char, char, "Char")
GET_STRINGLIST(get_string_list, is_matrix_type, is_int32_type, int32NDArray, int32_array_value, int32_t, int32_t, "Integer")
GET_STRINGLIST(get_string_list, is_matrix_type, is_int16_type, int16NDArray, int16_array_value, int16_t, int16_t, "Short")
GET_STRINGLIST(get_string_list, is_matrix_type, is_uint16_type, uint16NDArray, uint16_array_value, uint16_t, uint16_t, "Word")
#undef GET_STRINGLIST

void COctaveInterface::get_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* &attrs)
{
	attrs=NULL;
}

/** set functions - to pass data from shogun to Octave */

void COctaveInterface::set_int(int32_t scalar)
{
	octave_value o(scalar);
	set_arg_increment(o);
}

void COctaveInterface::set_real(float64_t scalar)
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
void COctaveInterface::function_name(const sg_type* vec, int32_t len)				\
{																				\
	oct_type mat=oct_type(dim_vector(1, len));									\
																				\
	for (int32_t i=0; i<len; i++)													\
			mat(i) = (if_type) vec[i];											\
																				\
	set_arg_increment(mat);														\
}
SET_VECTOR(set_vector, uint8NDArray, uint8_t, uint8_t, "Byte")
SET_VECTOR(set_vector, charMatrix, char, char, "Char")
SET_VECTOR(set_vector, int32NDArray, int32_t, int32_t, "Integer")
SET_VECTOR(set_vector, int16NDArray, int16_t, int16_t, "Short")
SET_VECTOR(set_vector, Matrix, float32_t, float32_t, "Single Precision")
SET_VECTOR(set_vector, Matrix, float64_t, float64_t, "Double Precision")
SET_VECTOR(set_vector, uint16NDArray, uint16_t, uint16_t, "Word")
#undef SET_VECTOR

#define SET_MATRIX(function_name, oct_type, sg_type, if_type, error_string)		\
void COctaveInterface::function_name(const sg_type* matrix, int32_t num_feat, int32_t num_vec) \
{																				\
	oct_type mat=oct_type(dim_vector(num_feat, num_vec));						\
																				\
	for (int32_t i=0; i<num_vec; i++)												\
	{																			\
		for (int32_t j=0; j<num_feat; j++)											\
			mat(j,i) = (if_type) matrix[j+i*num_feat];							\
	}																			\
																				\
	set_arg_increment(mat);														\
}
SET_MATRIX(set_matrix, uint8NDArray, uint8_t, uint8_t, "Byte")
SET_MATRIX(set_matrix, charMatrix, char, char, "Char")
SET_MATRIX(set_matrix, int32NDArray, int32_t, int32_t, "Integer")
SET_MATRIX(set_matrix, int16NDArray, int16_t, int16_t, "Short")
SET_MATRIX(set_matrix, Matrix, float32_t, float32_t, "Single Precision")
SET_MATRIX(set_matrix, Matrix, float64_t, float64_t, "Double Precision")
SET_MATRIX(set_matrix, uint16NDArray, uint16_t, uint16_t, "Word")
#undef SET_MATRIX

void COctaveInterface::set_sparse_matrix(const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec, int64_t nnz)
{
	SparseMatrix sm((octave_idx_type) num_feat, (octave_idx_type) num_vec, (octave_idx_type) nnz);

	int64_t offset=0;
	for (int32_t i=0; i<num_vec; i++)
	{
		int32_t len=matrix[i].num_feat_entries;
		sm.cidx(i)=offset;
		for (int32_t j=0; j<len; j++)
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
void COctaveInterface::function_name(const SGString<sg_type>* strings, int32_t num_str)	\
{																					\
	if (!strings)																	\
		SG_ERROR("Given strings are invalid.\n")									\
																					\
	Cell c= Cell(dim_vector(1,num_str));											\
	if (c.nelem()!=num_str)															\
		SG_ERROR("Couldn't create Cell Array of %d strings.\n", num_str)			\
																					\
	for (int32_t i=0; i<num_str; i++)												\
	{																				\
		int32_t len=strings[i].slen;												\
		if (len>0)																	\
		{																			\
			oct_type str(dim_vector(1,len));										\
			if (str.cols()!=len)													\
				SG_ERROR("Couldn't create " error_string " String %d of length %d.\n", i, len)	\
																					\
			for (int32_t j=0; j<len; j++)											\
				str(j)= (if_type) strings[i].string[j];								\
			c.elem(i)=str;															\
		}																			\
	}																				\
																					\
	set_arg_increment(c);															\
}
SET_STRINGLIST(set_string_list, int8NDArray, uint8_t, uint8_t, "Byte")
SET_STRINGLIST(set_string_list, charNDArray, char, char, "Char")
SET_STRINGLIST(set_string_list, int32NDArray, int32_t, int32_t, "Integer")
SET_STRINGLIST(set_string_list, int16NDArray, int16_t, int16_t, "Short")
SET_STRINGLIST(set_string_list, uint16NDArray, uint16_t, uint16_t, "Word")
#undef SET_STRINGLIST

void COctaveInterface::set_attribute_struct(const CDynamicArray<T_ATTRIBUTE>* attrs)
{
/*	octave_value arg=get_arg_increment();
	if (!arg.is_map())
		SG_ERROR("not a struct");
	attrs = new CDynamicArray<T_ATTRIBUTE>();*/
}

bool COctaveInterface::cmd_run_python()
{
#ifdef HAVE_PYTHON
	return CPythonInterface::run_python_helper(this);
#else
	return false;
#endif
}

bool COctaveInterface::cmd_run_r()
{
#ifdef HAVE_R
	return CRInterface::run_r_helper(this);
#else
	return false;
#endif
}

void COctaveInterface::recover_from_exception(void)
{
#if defined(OCTAVE_APIVERSION) && OCTAVE_APIVERSION < 37
  unwind_protect::run_all ();
#endif
  can_interrupt = true;
  octave_interrupt_immediately = 0;
  octave_interrupt_state = 0;

#if !defined(OCTAVE_APIVERSION) || OCTAVE_APIVERSION >= 37
  octave_exception_state = octave_no_exception;
#else
  octave_allocation_error = 0;
#endif
  octave_restore_signal_mask ();
  octave_catch_interrupts ();
}

void COctaveInterface::clear_octave_globals()
{
	//string_vector gvars = symbol_table::global_variable_names();

	//int gcount = gvars.length();

	//for (int i = 0; i < gcount; i++)
	//	symbol_table::clear_global(gvars[i]);
	int parse_status;
	eval_string("clear all", false, parse_status);
	//	symbol_table::clear_global_pattern ("*");
	//global_sym_tab->clear();
}

void COctaveInterface::run_octave_init()
{
	char* name=strdup("octave");
	char* opts=strdup("-q");
	char* argv[2]={name, opts};
	octave_main(2,argv,1);
	free(opts);
	free(name);
}

void COctaveInterface::run_octave_exit()
{
#if defined(OCTAVE_MAJOR_VERSION) && OCTAVE_MAJOR_VERSION >= 3 && defined(OCTAVE_MINOR_VERSION) && OCTAVE_MINOR_VERSION >= 8
	clean_up_and_exit (0); 
#else
	do_octave_atexit();
#endif
}

bool COctaveInterface::run_octave_helper(CSGInterface* from_if)
{
	SG_OBJ_DEBUG(from_if, "Entering Octave\n");
	octave_save_signal_mask ();

	if (octave_set_current_context)
	{
#if defined (USE_EXCEPTIONS_FOR_INTERRUPTS)
		panic_impossible ();
#else
#if defined(OCTAVE_APIVERSION) && OCTAVE_APIVERSION < 37
		unwind_protect::run_all ();
#endif
		raw_mode (0);
		octave_restore_signal_mask ();
#endif
	}

	can_interrupt = true;
	octave_catch_interrupts ();
	octave_initialized = true;

	try
	{
		int parse_status;
		char* octave_code=NULL;
		clear_octave_globals();

		for (int i=0; i<from_if->get_nrhs(); i++)
		{
			int len=0;
			char* var_name = from_if->get_string(len);
			SG_OBJ_DEBUG(from_if, "var_name = '%s'\n", var_name);
			if (strmatch(var_name, "octavecode"))
			{
				len=0;
				octave_code=from_if->get_string(len);
				SG_OBJ_DEBUG(from_if, "octave_code = '%s'\n", octave_code);
				break;
			}
			else
			{
				octave_value_list args;

				COctaveInterface* in = new COctaveInterface(args, 1, false);
				in->create_return_values(1);
				from_if->translate_arg(from_if, in);
#if !defined(OCTAVE_APIVERSION) || OCTAVE_APIVERSION >= 37
				symbol_table::varref (var_name) = in->get_return_values()(0);
#else
				set_global_value(var_name, in->get_return_values()(0));
#endif
				SG_FREE(var_name);
				SG_UNREF(in);
			}
		}

#if !defined(OCTAVE_APIVERSION) || OCTAVE_APIVERSION >= 37
#else
		symbol_table* old=curr_sym_tab;
		curr_sym_tab = global_sym_tab;
#endif
		reset_error_handler ();
		eval_string(octave_code, false, parse_status);
		SG_FREE(octave_code);

		int32_t sz=0;
		octave_value_list results;

#if !defined(OCTAVE_APIVERSION) || OCTAVE_APIVERSION >= 37
		if (symbol_table::is_variable("results"))
		{
			results = symbol_table::varval("results");
			//results = get_global_value("results", false);
			sz=results.length();
		}
#else
		if (curr_sym_tab->lookup("results"))
		{
			results = get_global_value("results", false);
			sz=results.length();
		}
#endif

		if (sz>0)
		{
			if (results(0).is_cs_list())
			{
				SG_OBJ_DEBUG(from_if, "Found return list of length %d\n", results(0).length());
				results=results(0).list_value();
				sz=results.length();
			}
		}

		if (sz>0 && from_if->create_return_values(sz))
		{
			SG_OBJ_DEBUG(from_if, "Found %d args\n", sz);
			COctaveInterface* out = new COctaveInterface(results, sz, false);

			//process d
			for (int32_t i=0; i<sz; i++)
				from_if->translate_arg(out, from_if);

			SG_UNREF(out);
		}
		else
		{
			if (sz!=from_if->get_nlhs())
			{
				SG_OBJ_ERROR(from_if, "Number of return values (%d) does not match number of expected"
						" return values (%d).\n", sz, from_if->get_nlhs());
			}
		}

#if !defined(OCTAVE_APIVERSION) || OCTAVE_APIVERSION >= 37
#else
		curr_sym_tab=old;
#endif
	}
	catch (octave_interrupt_exception)
	{
		recover_from_exception ();
		SG_SPRINT("%\n");
	}
	catch (std::bad_alloc)
	{
		recover_from_exception ();
		SG_SPRINT("%\n");
	}

	octave_restore_signal_mask();
	octave_initialized = false;

	SG_OBJ_DEBUG(from_if, "Leaving Octave.\n");
	return true;
}

#ifdef HAVE_ELWMS
DEFUN_DLD (elwms, prhs, nlhs, "shogun.")
#else
DEFUN_DLD (sg, prhs, nlhs, "shogun.")
#endif
{
	try
	{
		if (!interface)
		{
			// init_shogun has to be called before anything else
			// exit_shogun is called upon destruction of the interface (see
			// destructor of COctaveInterface
			init_shogun(&octave_print_message, &octave_print_warning,
					&octave_print_error, &octave_cancel_computations);
			interface=new COctaveInterface(prhs, nlhs);
#ifdef HAVE_PYTHON
			CPythonInterface::run_python_init();
#endif
#ifdef HAVE_R
			CRInterface::run_r_init();
#endif
		}
		else
			((COctaveInterface*) interface)->reset(prhs, nlhs);

		if (!interface->handle())
			SG_SERROR("Unknown command.\n");

		return ((COctaveInterface*) interface)->get_return_values();
	}
	catch (std::bad_alloc)
	{
		SG_SPRINT("Out of memory error.\n");
		return octave_value_list();
	}
	catch (ShogunException e)
	{
		error("%s", e.get_exception_string());
		return octave_value_list();
	}
	catch (...)
	{
		error("%s", "Returning from SHOGUN in error.");
		return octave_value_list();
	}
}

/* to be run on exiting matlab ... does not seem to be possible right now:
 * run_octave_exit()
 * run_python_exit()
 * run_r_exit()
 */
