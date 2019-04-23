/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Soeren Sonnenburg, Saloni Nigam, Sergey Lisitsyn,
 *          Viktor Gal
 */

#include <shogun/lib/memory.h>
#include <shogun/io/streaming/StreamingFile.h>
#include <fcntl.h>
#ifdef _WIN32
#include <io.h>
#endif

namespace shogun
{
/**
 * Dummy implementations of all the functions declared in the header
 * file.
 *
 * The derived class should reimplement whichever functions it
 * needs to use.
 *
 * If this is not done, the default implementation sets
 * the vector to NULL and number of features to -1.
 **/

/* For dense vectors */
#define GET_VECTOR(fname, conv, sg_type)				\
	void StreamingFile::get_vector					\
	(sg_type*& vector, int32_t& num_feat)				\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		error("Read function not supported by the feature type!"); \
	}

GET_VECTOR(get_bool_vector, atoi, bool)
GET_VECTOR(get_byte_vector, atoi, uint8_t)
GET_VECTOR(get_char_vector, atoi, char)
GET_VECTOR(get_int_vector, atoi, int32_t)
GET_VECTOR(get_shortreal_vector, atof, float32_t)
GET_VECTOR(get_real_vector, atof, float64_t)
GET_VECTOR(get_short_vector, atoi, int16_t)
GET_VECTOR(get_word_vector, atoi, uint16_t)
GET_VECTOR(get_int8_vector, atoi, int8_t)
GET_VECTOR(get_uint_vector, atoi, uint32_t)
GET_VECTOR(get_long_vector, atoi, int64_t)
GET_VECTOR(get_ulong_vector, atoi, uint64_t)
GET_VECTOR(get_longreal_vector, atoi, floatmax_t)
#undef GET_VECTOR

/* For dense vectors with labels */
#define GET_VECTOR_AND_LABEL(fname, conv, sg_type)			\
	void StreamingFile::get_vector_and_label			\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		error("Read function not supported by the feature type!"); \
	}

GET_VECTOR_AND_LABEL(get_bool_vector_and_label, str_to_bool, bool)
GET_VECTOR_AND_LABEL(get_byte_vector_and_label, atoi, uint8_t)
GET_VECTOR_AND_LABEL(get_char_vector_and_label, atoi, char)
GET_VECTOR_AND_LABEL(get_int_vector_and_label, atoi, int32_t)
GET_VECTOR_AND_LABEL(get_shortreal_vector_and_label, atof, float32_t)
GET_VECTOR_AND_LABEL(get_real_vector_and_label, atof, float64_t)
GET_VECTOR_AND_LABEL(get_short_vector_and_label, atoi, int16_t)
GET_VECTOR_AND_LABEL(get_word_vector_and_label, atoi, uint16_t)
GET_VECTOR_AND_LABEL(get_int8_vector_and_label, atoi, int8_t)
GET_VECTOR_AND_LABEL(get_uint_vector_and_label, atoi, uint32_t)
GET_VECTOR_AND_LABEL(get_long_vector_and_label, atoi, int64_t)
GET_VECTOR_AND_LABEL(get_ulong_vector_and_label, atoi, uint64_t)
GET_VECTOR_AND_LABEL(get_longreal_vector_and_label, atoi, floatmax_t)
#undef GET_VECTOR_AND_LABEL

/* For string vectors */
#define GET_STRING(fname, conv, sg_type)				\
	void StreamingFile::get_string					\
	(sg_type*& vector, int32_t& num_feat)				\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		error("Read function not supported by the feature type!"); \
	}

GET_STRING(get_bool_string, str_to_bool, bool)
GET_STRING(get_byte_string, atoi, uint8_t)
GET_STRING(get_char_string, atoi, char)
GET_STRING(get_int_string, atoi, int32_t)
GET_STRING(get_shortreal_string, atof, float32_t)
GET_STRING(get_real_string, atof, float64_t)
GET_STRING(get_short_string, atoi, int16_t)
GET_STRING(get_word_string, atoi, uint16_t)
GET_STRING(get_int8_string, atoi, int8_t)
GET_STRING(get_uint_string, atoi, uint32_t)
GET_STRING(get_long_string, atoi, int64_t)
GET_STRING(get_ulong_string, atoi, uint64_t)
GET_STRING(get_longreal_string, atoi, floatmax_t)
#undef GET_STRING

/* For string vectors with labels */
#define GET_STRING_AND_LABEL(fname, conv, sg_type)			\
	void StreamingFile::get_string_and_label			\
	(sg_type*& vector, int32_t& num_feat, float64_t& label)		\
	{								\
		vector=NULL;						\
		num_feat=-1;							\
		error("Read function not supported by the feature type!"); \
	}

GET_STRING_AND_LABEL(get_bool_string_and_label, str_to_bool, bool)
GET_STRING_AND_LABEL(get_byte_string_and_label, atoi, uint8_t)
GET_STRING_AND_LABEL(get_char_string_and_label, atoi, char)
GET_STRING_AND_LABEL(get_int_string_and_label, atoi, int32_t)
GET_STRING_AND_LABEL(get_shortreal_string_and_label, atof, float32_t)
GET_STRING_AND_LABEL(get_real_string_and_label, atof, float64_t)
GET_STRING_AND_LABEL(get_short_string_and_label, atoi, int16_t)
GET_STRING_AND_LABEL(get_word_string_and_label, atoi, uint16_t)
GET_STRING_AND_LABEL(get_int8_string_and_label, atoi, int8_t)
GET_STRING_AND_LABEL(get_uint_string_and_label, atoi, uint32_t)
GET_STRING_AND_LABEL(get_long_string_and_label, atoi, int64_t)
GET_STRING_AND_LABEL(get_ulong_string_and_label, atoi, uint64_t)
GET_STRING_AND_LABEL(get_longreal_string_and_label, atoi, floatmax_t)
#undef GET_STRING_AND_LABEL

/* For sparse vectors */
#define GET_SPARSE_VECTOR(fname, conv, sg_type)				\
									\
	void StreamingFile::get_sparse_vector				\
	(SGSparseVectorEntry<sg_type>*& vector, int32_t& num_feat)	\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		error("Read function not supported by the feature type!"); \
	}

GET_SPARSE_VECTOR(get_bool_sparse_vector, str_to_bool, bool)
GET_SPARSE_VECTOR(get_byte_sparse_vector, atoi, uint8_t)
GET_SPARSE_VECTOR(get_char_sparse_vector, atoi, char)
GET_SPARSE_VECTOR(get_int_sparse_vector, atoi, int32_t)
GET_SPARSE_VECTOR(get_shortreal_sparse_vector, atof, float32_t)
GET_SPARSE_VECTOR(get_real_sparse_vector, atof, float64_t)
GET_SPARSE_VECTOR(get_short_sparse_vector, atoi, int16_t)
GET_SPARSE_VECTOR(get_word_sparse_vector, atoi, uint16_t)
GET_SPARSE_VECTOR(get_int8_sparse_vector, atoi, int8_t)
GET_SPARSE_VECTOR(get_uint_sparse_vector, atoi, uint32_t)
GET_SPARSE_VECTOR(get_long_sparse_vector, atoi, int64_t)
GET_SPARSE_VECTOR(get_ulong_sparse_vector, atoi, uint64_t)
GET_SPARSE_VECTOR(get_longreal_sparse_vector, atoi, floatmax_t)
#undef GET_SPARSE_VECTOR

/* For sparse vectors with labels */
#define GET_SPARSE_VECTOR_AND_LABEL(fname, conv, sg_type)		\
									\
	void StreamingFile::get_sparse_vector_and_label		\
	(SGSparseVectorEntry<sg_type>*& vector,				\
	 int32_t& num_feat,						\
	 float64_t& label)						\
	{								\
		vector=NULL;						\
		num_feat=-1;						\
		error("Read function not supported by the feature type!"); \
	}

GET_SPARSE_VECTOR_AND_LABEL(get_bool_sparse_vector_and_label, str_to_bool, bool)
GET_SPARSE_VECTOR_AND_LABEL(get_byte_sparse_vector_and_label, atoi, uint8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_char_sparse_vector_and_label, atoi, char)
GET_SPARSE_VECTOR_AND_LABEL(get_int_sparse_vector_and_label, atoi, int32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_shortreal_sparse_vector_and_label, atof, float32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_real_sparse_vector_and_label, atof, float64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_short_sparse_vector_and_label, atoi, int16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_word_sparse_vector_and_label, atoi, uint16_t)
GET_SPARSE_VECTOR_AND_LABEL(get_int8_sparse_vector_and_label, atoi, int8_t)
GET_SPARSE_VECTOR_AND_LABEL(get_uint_sparse_vector_and_label, atoi, uint32_t)
GET_SPARSE_VECTOR_AND_LABEL(get_long_sparse_vector_and_label, atoi, int64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_ulong_sparse_vector_and_label, atoi, uint64_t)
GET_SPARSE_VECTOR_AND_LABEL(get_longreal_sparse_vector_and_label, atoi, floatmax_t)
#undef GET_SPARSE_VECTOR_AND_LABEL

}

using namespace shogun;

StreamingFile::StreamingFile() : SGObject()
{
	buf=NULL;
	filename=NULL;
}

StreamingFile::StreamingFile(const char* fname, char rw) : SGObject()
{
	task=rw;
	filename=get_strdup(fname);
	int mode = O_LARGEFILE;

	switch (rw)
	{
	case 'r':
		mode |= O_RDONLY;
		break;
	case 'w':
		mode |= O_WRONLY;
		break;
	default:
		error("Unknown mode '{}'", task);
	}

	if (filename)
	{
		int file = open((const char*) filename, mode);
		if (file < 0)
			error("Error opening file '{}'", filename);

		buf = std::make_shared<IOBuffer>(file);
	}
	else
		error("Error getting the file name!");
}

StreamingFile::~StreamingFile()
{
	SG_FREE(filename);
}
