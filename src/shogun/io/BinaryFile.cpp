/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Thoralf Klein, Sergey Lisitsyn, Heiko Strathmann,
 *          Chiyuan Zhang, Viktor Gal, Bjoern Esser
 */

#include <shogun/io/File.h>
#include <shogun/io/BinaryFile.h>
#include <shogun/io/SGIO.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGSparseVector.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

BinaryFile::BinaryFile()
{
	unstable(SOURCE_LOCATION);
}

BinaryFile::BinaryFile(FILE* f, const char* name) : File(f, name)
{
}

BinaryFile::BinaryFile(const char* fname, char rw, const char* name) : File(fname, rw, name)
{
}

BinaryFile::~BinaryFile()
{
}

#define GET_VECTOR(fname, sg_type, datatype)										\
void BinaryFile::fname(sg_type*& vec, int32_t& len)								\
{																					\
	if (!file)																		\
		error("File invalid.");												\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			            \
	if (dtype!=datatype)															\
		error("Datatype mismatch");											\
																					\
	if (fread(&len, sizeof(int32_t), 1, file)!=1)									\
		error("Failed to read vector length");									\
	vec=SG_MALLOC(sg_type, len);															\
	if (fread(vec, sizeof(sg_type), len, file)!=(size_t) len)						\
		error("Failed to read Matrix");										\
}

GET_VECTOR(get_vector, int8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT8))
GET_VECTOR(get_vector, uint8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT8))
GET_VECTOR(get_vector, char, TSGDataType(CT_VECTOR, ST_NONE, PT_CHAR))
GET_VECTOR(get_vector, int32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_VECTOR(get_vector, uint32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT32))
GET_VECTOR(get_vector, float32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT32))
GET_VECTOR(get_vector, float64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT64))
GET_VECTOR(get_vector, floatmax_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOATMAX))
GET_VECTOR(get_vector, int16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_VECTOR(get_vector, uint16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_VECTOR(get_vector, int64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT64))
GET_VECTOR(get_vector, uint64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT64))
#undef GET_VECTOR

#define GET_MATRIX(fname, sg_type, datatype)										\
void BinaryFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)		\
{																					\
	if (!file)																		\
		error("File invalid.");												\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			            \
	if (dtype!=datatype)															\
		error("Datatype mismatch");											\
																					\
	if (fread(&num_feat, sizeof(int32_t), 1, file)!=1 ||							\
			fread(&num_vec, sizeof(int32_t), 1, file)!=1)							\
		error("Failed to read Matrix dimensions");								\
	matrix=SG_MALLOC(sg_type, int64_t(num_feat)*num_vec);									\
	if (fread(matrix, sizeof(sg_type)*num_feat, num_vec, file)!=(size_t) num_vec)	\
		error("Failed to read Matrix");										\
}

GET_MATRIX(get_matrix, char, TSGDataType(CT_MATRIX, ST_NONE, PT_CHAR))
GET_MATRIX(get_matrix, uint8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_UINT8))
GET_MATRIX(get_matrix, int8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT8))
GET_MATRIX(get_matrix, int32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_matrix, uint32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_matrix, int64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_matrix, uint64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_matrix, int16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_matrix, uint16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_matrix, float32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT32))
GET_MATRIX(get_matrix, float64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT64))
GET_MATRIX(get_matrix, floatmax_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef GET_MATRIX

#define GET_NDARRAY(fname,sg_type,datatype)									\
void BinaryFile::fname(sg_type *& array, int32_t *& dims,int32_t & num_dims)\
{																			\
	size_t total = 1;														\
																			\
	if (!file)																\
		error("File invalid.");										\
																			\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL);							\
	read_header(&dtype);													\
																			\
	if (dtype!=datatype)													\
		error("Datatype mismatch");									\
																			\
	if (fread(&num_dims,sizeof(int32_t),1,file) != 1)						\
		error("Failed to read number of dimensions");					\
																			\
	dims = SG_MALLOC(int32_t, num_dims);											\
	if (fread(dims,sizeof(int32_t),num_dims,file) != (size_t)num_dims)		\
		error("Failed to read sizes of dimensions!");					\
																			\
	for (int32_t i = 0;i < num_dims;i++)									\
		total *= dims[i];													\
																			\
	array = SG_MALLOC(sg_type, total);												\
	if (fread(array,sizeof(sg_type),total,file) != (size_t)total)			\
		error("Failed to read array data!");								\
}

GET_NDARRAY(get_ndarray,uint8_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_UINT8));
GET_NDARRAY(get_ndarray,char,TSGDataType(CT_NDARRAY, ST_NONE, PT_CHAR));
GET_NDARRAY(get_ndarray,int32_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_INT32));
GET_NDARRAY(get_ndarray,int16_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_INT16));
GET_NDARRAY(get_ndarray,uint16_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_UINT16));
GET_NDARRAY(get_ndarray,float32_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_FLOAT32));
GET_NDARRAY(get_ndarray,float64_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_FLOAT64));
#undef GET_NDARRAY

#define GET_SPARSEMATRIX(fname, sg_type, datatype)										\
void BinaryFile::fname(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																						\
	if (!(file))																		\
		error("File invalid.");													\
																						\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			                \
	if (dtype!=datatype)																\
		error("Datatype mismatch");												\
																						\
	if (fread(&num_vec, sizeof(int32_t), 1, file)!=1)									\
		error("Failed to read number of vectors");									\
																						\
	matrix=SG_MALLOC(SGSparseVector<sg_type>, num_vec);												\
																						\
	for (int32_t i=0; i<num_vec; i++)													\
	{																					\
		new (&matrix[i]) SGSparseVector<sg_type>();										\
		int32_t len=0;																	\
		if (fread(&len, sizeof(int32_t), 1, file)!=1)									\
			error("Failed to read sparse vector length of vector idx={}", i);		\
		matrix[i].num_feat_entries=len;													\
		SGSparseVectorEntry<sg_type>* vec = SG_MALLOC(SGSparseVectorEntry<sg_type>, len);					\
		if (fread(vec, sizeof(SGSparseVectorEntry<sg_type>), len, file)!= (size_t) len)		\
			error("Failed to read sparse vector {}", i);							\
		matrix[i].features=vec;															\
		num_feat = Math::max(num_feat, matrix[i].get_num_dimensions()); \
	}																					\
}
GET_SPARSEMATRIX(get_sparse_matrix, bool, TSGDataType(CT_MATRIX, ST_NONE, PT_BOOL))
GET_SPARSEMATRIX(get_sparse_matrix, char, TSGDataType(CT_MATRIX, ST_NONE, PT_CHAR))
GET_SPARSEMATRIX(get_sparse_matrix, uint8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_UINT8))
GET_SPARSEMATRIX(get_sparse_matrix, int8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT8))
GET_SPARSEMATRIX(get_sparse_matrix, int32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_SPARSEMATRIX(get_sparse_matrix, uint32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_SPARSEMATRIX(get_sparse_matrix, int64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_SPARSEMATRIX(get_sparse_matrix, uint64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_SPARSEMATRIX(get_sparse_matrix, int16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_SPARSEMATRIX(get_sparse_matrix, uint16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_SPARSEMATRIX(get_sparse_matrix, float32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT32))
GET_SPARSEMATRIX(get_sparse_matrix, float64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT64))
GET_SPARSEMATRIX(get_sparse_matrix, floatmax_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef GET_SPARSEMATRIX


#define GET_STRING_LIST(fname, sg_type, datatype)												\
void BinaryFile::fname(SGVector<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len) \
{																								\
	strings=NULL;																				\
	num_str=0;																					\
	max_string_len=0;																			\
																								\
	if (!file)																					\
		error("File invalid.");															\
																								\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			                        \
	if (dtype!=datatype)																		\
		error("Datatype mismatch");														\
																								\
	if (fread(&num_str, sizeof(int32_t), 1, file)!=1)											\
		error("Failed to read number of strings");											\
																								\
	strings=SG_MALLOC(SGVector<sg_type>, num_str);														\
																								\
	for (int32_t i=0; i<num_str; i++)															\
	{																							\
		int32_t len=0;																			\
		if (fread(&len, sizeof(int32_t), 1, file)!=1)											\
			error("Failed to read string length of string with idx={}", i);					\
		strings[i] = SGVector<sg_type>(len);													\
		if (fread(strings[i].vector, sizeof(sg_type), len, file)!= (size_t) len)				\
			error("Failed to read string {}", i);												\
	}																							\
}

GET_STRING_LIST(get_string_list, char, TSGDataType(CT_VECTOR, ST_NONE, PT_CHAR))
GET_STRING_LIST(get_string_list, uint8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT8))
GET_STRING_LIST(get_string_list, int8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT8))
GET_STRING_LIST(get_string_list, int32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_STRING_LIST(get_string_list, uint32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_STRING_LIST(get_string_list, int64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT64))
GET_STRING_LIST(get_string_list, uint64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT64))
GET_STRING_LIST(get_string_list, int16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_STRING_LIST(get_string_list, uint16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_STRING_LIST(get_string_list, float32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT32))
GET_STRING_LIST(get_string_list, float64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT64))
GET_STRING_LIST(get_string_list, floatmax_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOATMAX))
#undef GET_STRING_LIST

/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, sg_type, dtype)							\
void BinaryFile::fname(const sg_type* vec, int32_t len)			\
{																	\
	if (!(file && vec))												\
		error("File or vector invalid.");						\
																	\
	TSGDataType t dtype; write_header(&t);						    \
																	\
	if (fwrite(&len, sizeof(int32_t), 1, file)!=1 ||				\
			fwrite(vec, sizeof(sg_type), len, file)!=(size_t) len)	\
		error("Failed to write vector");						\
}
SET_VECTOR(set_vector, int8_t, (CT_VECTOR, ST_NONE, PT_INT8))
SET_VECTOR(set_vector, uint8_t, (CT_VECTOR, ST_NONE, PT_UINT8))
SET_VECTOR(set_vector, char, (CT_VECTOR, ST_NONE, PT_CHAR))
SET_VECTOR(set_vector, int32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_VECTOR(set_vector, uint32_t, (CT_VECTOR, ST_NONE, PT_UINT32))
SET_VECTOR(set_vector, float32_t, (CT_VECTOR, ST_NONE, PT_FLOAT32))
SET_VECTOR(set_vector, float64_t, (CT_VECTOR, ST_NONE, PT_FLOAT64))
SET_VECTOR(set_vector, floatmax_t, (CT_VECTOR, ST_NONE, PT_FLOATMAX))
SET_VECTOR(set_vector, int16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_VECTOR(set_vector, uint16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_VECTOR(set_vector, int64_t, (CT_VECTOR, ST_NONE, PT_INT64))
SET_VECTOR(set_vector, uint64_t, (CT_VECTOR, ST_NONE, PT_UINT64))
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, dtype) \
void BinaryFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																					\
	if (!(file && matrix))															\
		error("File or matrix invalid.");										\
																					\
	TSGDataType t dtype; write_header(&t);						                    \
																					\
	if (fwrite(&num_feat, sizeof(int32_t), 1, file)!=1 ||							\
			fwrite(&num_vec, sizeof(int32_t), 1, file)!=1 ||						\
			fwrite(matrix, sizeof(sg_type)*num_feat, num_vec, file)!=(size_t) num_vec)	\
		error("Failed to write Matrix");										\
}
SET_MATRIX(set_matrix, char, (CT_MATRIX, ST_NONE, PT_CHAR))
SET_MATRIX(set_matrix, uint8_t, (CT_MATRIX, ST_NONE, PT_UINT8))
SET_MATRIX(set_matrix, int8_t, (CT_MATRIX, ST_NONE, PT_INT8))
SET_MATRIX(set_matrix, int32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_MATRIX(set_matrix, uint32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_MATRIX(set_matrix, int64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_MATRIX(set_matrix, uint64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_MATRIX(set_matrix, int16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_MATRIX(set_matrix, uint16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_MATRIX(set_matrix, float32_t, (CT_MATRIX, ST_NONE, PT_FLOAT32))
SET_MATRIX(set_matrix, float64_t, (CT_MATRIX, ST_NONE, PT_FLOAT64))
SET_MATRIX(set_matrix, floatmax_t, (CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef SET_MATRIX

#define SET_NDARRAY(fname,sg_type,datatype)									\
void BinaryFile::fname(const sg_type * array, int32_t * dims,int32_t num_dims)	\
{																			\
	size_t total = 1;														\
																			\
	if (!file)																\
		error("File invalid.");										\
																			\
	if (!array)																\
		error("Invalid array!");										\
																			\
	TSGDataType t datatype;													\
	write_header(&t);														\
																			\
	if (fwrite(&num_dims,sizeof(int32_t),1,file) != 1)						\
		error("Failed to write number of dimensions!");				\
																			\
	if (fwrite(dims,sizeof(int32_t),num_dims,file) != (size_t)num_dims)		\
		error("Failed to write sizes of dimensions!");					\
																			\
	for (int32_t i = 0;i < num_dims;i++)										\
		total *= dims[i];													\
																			\
	if (fwrite(array,sizeof(sg_type),total,file) != (size_t)total)			\
		error("Failed to write array data!");							\
}

SET_NDARRAY(set_ndarray,uint8_t,(CT_NDARRAY, ST_NONE, PT_UINT8));
SET_NDARRAY(set_ndarray,char,(CT_NDARRAY, ST_NONE, PT_CHAR));
SET_NDARRAY(set_ndarray,int32_t,(CT_NDARRAY, ST_NONE, PT_INT32));
SET_NDARRAY(set_ndarray,int16_t,(CT_NDARRAY, ST_NONE, PT_INT16));
SET_NDARRAY(set_ndarray,uint16_t,(CT_NDARRAY, ST_NONE, PT_UINT16));
SET_NDARRAY(set_ndarray,float32_t,(CT_NDARRAY, ST_NONE, PT_FLOAT32));
SET_NDARRAY(set_ndarray,float64_t,(CT_NDARRAY, ST_NONE, PT_FLOAT64));
#undef SET_NDARRAY

#define SET_SPARSEMATRIX(fname, sg_type, dtype)			\
void BinaryFile::fname(const SGSparseVector<sg_type>* matrix,	\
		int32_t num_feat, int32_t num_vec)					\
{															\
	if (!(file && matrix))									\
		error("File or matrix invalid.");				\
															\
	TSGDataType t dtype; write_header(&t);					\
															\
	if (fwrite(&num_vec, sizeof(int32_t), 1, file)!=1)		\
		error("Failed to write Sparse Matrix");		\
															\
	for (int32_t i=0; i<num_vec; i++)						\
	{														\
		SGSparseVectorEntry<sg_type>* vec = matrix[i].features;	\
		int32_t len=matrix[i].num_feat_entries;				\
		if ((fwrite(&len, sizeof(int32_t), 1, file)!=1) ||	\
				(fwrite(vec, sizeof(SGSparseVectorEntry<sg_type>), len, file)!= (size_t) len))		\
			error("Failed to write Sparse Matrix");	\
	}														\
}
SET_SPARSEMATRIX(set_sparse_matrix, bool, (CT_MATRIX, ST_NONE, PT_BOOL))
SET_SPARSEMATRIX(set_sparse_matrix, char, (CT_MATRIX, ST_NONE, PT_CHAR))
SET_SPARSEMATRIX(set_sparse_matrix, uint8_t, (CT_MATRIX, ST_NONE, PT_UINT8))
SET_SPARSEMATRIX(set_sparse_matrix, int8_t, (CT_MATRIX, ST_NONE, PT_INT8))
SET_SPARSEMATRIX(set_sparse_matrix, int32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_SPARSEMATRIX(set_sparse_matrix, uint32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_SPARSEMATRIX(set_sparse_matrix, int64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_SPARSEMATRIX(set_sparse_matrix, uint64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_SPARSEMATRIX(set_sparse_matrix, int16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_SPARSEMATRIX(set_sparse_matrix, uint16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_SPARSEMATRIX(set_sparse_matrix, float32_t, (CT_MATRIX, ST_NONE, PT_FLOAT32))
SET_SPARSEMATRIX(set_sparse_matrix, float64_t, (CT_MATRIX, ST_NONE, PT_FLOAT64))
SET_SPARSEMATRIX(set_sparse_matrix, floatmax_t, (CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef SET_SPARSEMATRIX

#define SET_STRING_LIST(fname, sg_type, dtype) \
void BinaryFile::fname(const SGVector<sg_type>* strings, int32_t num_str)	\
{																						\
	if (!(file && strings))																\
		error("File or strings invalid.");											\
																						\
	TSGDataType t dtype; write_header(&t);								                \
	for (int32_t i=0; i<num_str; i++)													\
	{																					\
		int32_t len = strings[i].vlen;												\
		if ((fwrite(&len, sizeof(int32_t), 1, file)!=1) ||								\
				(fwrite(strings[i].vector, sizeof(sg_type), len, file)!= (size_t) len))	\
			error("Failed to write Sparse Matrix");								\
	}																					\
}
SET_STRING_LIST(set_string_list, char, (CT_VECTOR, ST_NONE, PT_CHAR))
SET_STRING_LIST(set_string_list, uint8_t, (CT_VECTOR, ST_NONE, PT_UINT8))
SET_STRING_LIST(set_string_list, int8_t, (CT_VECTOR, ST_NONE, PT_INT8))
SET_STRING_LIST(set_string_list, int32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_STRING_LIST(set_string_list, uint32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_STRING_LIST(set_string_list, int64_t, (CT_VECTOR, ST_NONE, PT_INT64))
SET_STRING_LIST(set_string_list, uint64_t, (CT_VECTOR, ST_NONE, PT_INT64))
SET_STRING_LIST(set_string_list, int16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_STRING_LIST(set_string_list, uint16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_STRING_LIST(set_string_list, float32_t, (CT_VECTOR, ST_NONE, PT_FLOAT32))
SET_STRING_LIST(set_string_list, float64_t, (CT_VECTOR, ST_NONE, PT_FLOAT64))
SET_STRING_LIST(set_string_list, floatmax_t, (CT_VECTOR, ST_NONE, PT_FLOATMAX))
#undef SET_STRING_LIST


int32_t BinaryFile::parse_first_header(TSGDataType& type)
{
	    return -1;
}

int32_t BinaryFile::parse_next_header(TSGDataType& type)
{
	    return -1;
}

void
BinaryFile::read_header(TSGDataType* dest)
{
	ASSERT(file)
	ASSERT(dest)

	if (fseek(file, 0L, SEEK_SET)!=0)
		error("Error seeking file '{}' to the beginning.", filename);

	char fourcc[4];
	uint16_t endian=0;

	if (fread(&fourcc, sizeof(char), 4, file)!=4)
		error("Error reading fourcc header in file '{}'", filename);

	if (fread(&endian, sizeof(uint16_t), 1, file)!=1)
		error("Error reading endian header in file '{}'", filename);

	if ((fread(&dest->m_ctype, sizeof(dest->m_ctype), 1, file)!=1) ||
			(fread(&dest->m_ptype, sizeof(dest->m_ptype), 1, file)!=1))
		error("Error reading datatype header in file '{}'", filename);

	if (strncmp(fourcc, "SG01", 4))
		error("Header mismatch, expected SG01 in file '{}'", filename);
}

void
BinaryFile::write_header(const TSGDataType* datatype)
{
	ASSERT(file)

	const char* fourcc="SG01";
	uint16_t endian=0x1234;

	if (!((fwrite(fourcc, sizeof(char), 4, file)==4) &&
		  (fwrite(&endian, sizeof(uint16_t), 1, file)==1) &&
		  (fwrite(&datatype->m_ctype, sizeof(datatype->m_ctype), 1,
				  file)==1)
		  && (fwrite(&datatype->m_ptype, sizeof(datatype->m_ptype), 1,
					 file)==1)
			))
		error("Error writing header");
}
