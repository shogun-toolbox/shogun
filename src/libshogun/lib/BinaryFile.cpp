/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#include "lib/File.h"
#include "features/SparseFeatures.h"
#include "lib/BinaryFile.h"

using namespace shogun;

CBinaryFile::CBinaryFile(void)
{
	SG_UNSTABLE("CBinaryFile::CBinaryFile(void)", "\n");
}

CBinaryFile::CBinaryFile(FILE* f, const char* name) : CFile(f, name)
{
}

CBinaryFile::CBinaryFile(char* fname, char rw, const char* name) : CFile(fname, rw, name)
{
}

CBinaryFile::~CBinaryFile()
{
}

#define GET_VECTOR(fname, sg_type, datatype)										\
void CBinaryFile::fname(sg_type*& vec, int32_t& len)								\
{																					\
	if (!file)																		\
		SG_ERROR("File invalid.\n");												\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			            \
	if (dtype!=datatype)															\
		SG_ERROR("Datatype mismatch\n");											\
																					\
	if (fread(&len, sizeof(int32_t), 1, file)!=1)									\
		SG_ERROR("Failed to read vector length\n");									\
	vec=new sg_type[len];															\
	if (fread(vec, sizeof(sg_type), len, file)!=(size_t) len)						\
		SG_ERROR("Failed to read Matrix\n");										\
}

GET_VECTOR(get_byte_vector, uint8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT8))
GET_VECTOR(get_char_vector, char, TSGDataType(CT_VECTOR, ST_NONE, PT_CHAR))
GET_VECTOR(get_int_vector, int32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_VECTOR(get_shortreal_vector, float32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT32))
GET_VECTOR(get_real_vector, float64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT64))
GET_VECTOR(get_short_vector, int16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_VECTOR(get_word_vector, uint16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
#undef GET_VECTOR

#define GET_MATRIX(fname, sg_type, datatype)										\
void CBinaryFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)		\
{																					\
	if (!file)																		\
		SG_ERROR("File invalid.\n");												\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			            \
	if (dtype!=datatype)															\
		SG_ERROR("Datatype mismatch\n");											\
																					\
	if (fread(&num_feat, sizeof(int32_t), 1, file)!=1 ||							\
			fread(&num_vec, sizeof(int32_t), 1, file)!=1)							\
		SG_ERROR("Failed to read Matrix dimensions\n");								\
	matrix=new sg_type[int64_t(num_feat)*num_vec];									\
	if (fread(matrix, sizeof(sg_type)*num_feat, num_vec, file)!=(size_t) num_vec)	\
		SG_ERROR("Failed to read Matrix\n");										\
}

GET_MATRIX(get_char_matrix, char, TSGDataType(CT_MATRIX, ST_NONE, PT_CHAR))
GET_MATRIX(get_byte_matrix, uint8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_UINT8))
GET_MATRIX(get_int8_matrix, int8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT8))
GET_MATRIX(get_int_matrix, int32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_uint_matrix, uint32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_MATRIX(get_long_matrix, int64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_ulong_matrix, uint64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_MATRIX(get_short_matrix, int16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_word_matrix, uint16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_MATRIX(get_shortreal_matrix, float32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT32))
GET_MATRIX(get_real_matrix, float64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT64))
GET_MATRIX(get_longreal_matrix, floatmax_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef GET_MATRIX

#define GET_NDARRAY(fname,sg_type,datatype)									\
void CBinaryFile::fname(sg_type *& array, int32_t *& dims,int32_t & num_dims)\
{																			\
	size_t total = 1;														\
																			\
	if (!file)																\
		SG_ERROR("File invalid.\n");										\
																			\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL);							\
	read_header(&dtype);													\
																			\
	if (dtype!=datatype)													\
		SG_ERROR("Datatype mismatch\n");									\
																			\
	if (fread(&num_dims,sizeof(int32_t),1,file) != 1)						\
		SG_ERROR("Failed to read number of dimensions");					\
																			\
	dims = new int32_t[num_dims];											\
	if (fread(dims,sizeof(int32_t),num_dims,file) != (size_t)num_dims)		\
		SG_ERROR("Failed to read sizes of dimensions!");					\
																			\
	for (int32_t i = 0;i < num_dims;i++)									\
		total *= dims[i];													\
																			\
	array = new sg_type[total];												\
	if (fread(array,sizeof(sg_type),total,file) != (size_t)total)			\
		SG_ERROR("Failed to read array data!");								\
}

GET_NDARRAY(get_byte_ndarray,uint8_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_UINT8));
GET_NDARRAY(get_char_ndarray,char,TSGDataType(CT_NDARRAY, ST_NONE, PT_CHAR));
GET_NDARRAY(get_int_ndarray,int32_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_INT32));
GET_NDARRAY(get_short_ndarray,int16_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_INT16));
GET_NDARRAY(get_word_ndarray,uint16_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_UINT16));
GET_NDARRAY(get_shortreal_ndarray,float32_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_FLOAT32));
GET_NDARRAY(get_real_ndarray,float64_t,TSGDataType(CT_NDARRAY, ST_NONE, PT_FLOAT64));
#undef GET_NDARRAY

#define GET_SPARSEMATRIX(fname, sg_type, datatype)										\
void CBinaryFile::fname(SGSparseMatrix<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																						\
	if (!(file))																		\
		SG_ERROR("File invalid.\n");													\
																						\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			                \
	if (dtype!=datatype)																\
		SG_ERROR("Datatype mismatch\n");												\
																						\
	if (fread(&num_vec, sizeof(int32_t), 1, file)!=1)									\
		SG_ERROR("Failed to read number of vectors\n");									\
																						\
	matrix=new SGSparseMatrix<sg_type>[num_vec];												\
																						\
	for (int32_t i=0; i<num_vec; i++)													\
	{																					\
		int32_t len=0;																	\
		if (fread(&len, sizeof(int32_t), 1, file)!=1)									\
			SG_ERROR("Failed to read sparse vector length of vector idx=%d\n", i);		\
		matrix[i].num_feat_entries=len;													\
		SGSparseMatrixEntry<sg_type>* vec = new SGSparseMatrixEntry<sg_type>[len];					\
		if (fread(vec, sizeof(SGSparseMatrixEntry<sg_type>), len, file)!= (size_t) len)		\
			SG_ERROR("Failed to read sparse vector %d\n", i);							\
		matrix[i].features=vec;															\
	}																					\
}
GET_SPARSEMATRIX(get_bool_sparsematrix, bool, TSGDataType(CT_MATRIX, ST_NONE, PT_BOOL))
GET_SPARSEMATRIX(get_char_sparsematrix, char, TSGDataType(CT_MATRIX, ST_NONE, PT_CHAR))
GET_SPARSEMATRIX(get_byte_sparsematrix, uint8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_UINT8))
GET_SPARSEMATRIX(get_int8_sparsematrix, int8_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT8))
GET_SPARSEMATRIX(get_int_sparsematrix, int32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_SPARSEMATRIX(get_uint_sparsematrix, uint32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT32))
GET_SPARSEMATRIX(get_long_sparsematrix, int64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_SPARSEMATRIX(get_ulong_sparsematrix, uint64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT64))
GET_SPARSEMATRIX(get_short_sparsematrix, int16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_SPARSEMATRIX(get_word_sparsematrix, uint16_t, TSGDataType(CT_MATRIX, ST_NONE, PT_INT16))
GET_SPARSEMATRIX(get_shortreal_sparsematrix, float32_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT32))
GET_SPARSEMATRIX(get_real_sparsematrix, float64_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOAT64))
GET_SPARSEMATRIX(get_longreal_sparsematrix, floatmax_t, TSGDataType(CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef GET_SPARSEMATRIX


#define GET_STRING_LIST(fname, sg_type, datatype)												\
void CBinaryFile::fname(SGString<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len) \
{																								\
	strings=NULL;																				\
	num_str=0;																					\
	max_string_len=0;																			\
																								\
	if (!file)																					\
		SG_ERROR("File invalid.\n");															\
																								\
	TSGDataType dtype(CT_SCALAR, ST_NONE, PT_BOOL); read_header(&dtype);			                        \
	if (dtype!=datatype)																		\
		SG_ERROR("Datatype mismatch\n");														\
																								\
	if (fread(&num_str, sizeof(int32_t), 1, file)!=1)											\
		SG_ERROR("Failed to read number of strings\n");											\
																								\
	strings=new SGString<sg_type>[num_str];														\
																								\
	for (int32_t i=0; i<num_str; i++)															\
	{																							\
		int32_t len=0;																			\
		if (fread(&len, sizeof(int32_t), 1, file)!=1)											\
			SG_ERROR("Failed to read string length of string with idx=%d\n", i);				\
		strings[i].length=len;																	\
		sg_type* str = new sg_type[len];														\
		if (fread(str, sizeof(sg_type), len, file)!= (size_t) len)								\
			SG_ERROR("Failed to read string %d\n", i);											\
		strings[i].string=str;																	\
	}																							\
}

GET_STRING_LIST(get_char_string_list, char, TSGDataType(CT_VECTOR, ST_NONE, PT_CHAR))
GET_STRING_LIST(get_byte_string_list, uint8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_UINT8))
GET_STRING_LIST(get_int8_string_list, int8_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT8))
GET_STRING_LIST(get_int_string_list, int32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_STRING_LIST(get_uint_string_list, uint32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT32))
GET_STRING_LIST(get_long_string_list, int64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT64))
GET_STRING_LIST(get_ulong_string_list, uint64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT64))
GET_STRING_LIST(get_short_string_list, int16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_STRING_LIST(get_word_string_list, uint16_t, TSGDataType(CT_VECTOR, ST_NONE, PT_INT16))
GET_STRING_LIST(get_shortreal_string_list, float32_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT32))
GET_STRING_LIST(get_real_string_list, float64_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOAT64))
GET_STRING_LIST(get_longreal_string_list, floatmax_t, TSGDataType(CT_VECTOR, ST_NONE, PT_FLOATMAX))
#undef GET_STRING_LIST

/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, sg_type, dtype)							\
void CBinaryFile::fname(const sg_type* vec, int32_t len)			\
{																	\
	if (!(file && vec))												\
		SG_ERROR("File or vector invalid.\n");						\
																	\
	TSGDataType t dtype; write_header(&t);						    \
																	\
	if (fwrite(&len, sizeof(int32_t), 1, file)!=1 ||				\
			fwrite(vec, sizeof(sg_type), len, file)!=(size_t) len)	\
		SG_ERROR("Failed to write vector\n");						\
}
SET_VECTOR(set_byte_vector, uint8_t, (CT_VECTOR, ST_NONE, PT_UINT8))
SET_VECTOR(set_char_vector, char, (CT_VECTOR, ST_NONE, PT_CHAR))
SET_VECTOR(set_int_vector, int32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_VECTOR(set_shortreal_vector, float32_t, (CT_VECTOR, ST_NONE, PT_FLOAT32))
SET_VECTOR(set_real_vector, float64_t, (CT_VECTOR, ST_NONE, PT_FLOAT64))
SET_VECTOR(set_short_vector, int16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_VECTOR(set_word_vector, uint16_t, (CT_VECTOR, ST_NONE, PT_INT16))
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, dtype) \
void CBinaryFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																					\
	if (!(file && matrix))															\
		SG_ERROR("File or matrix invalid.\n");										\
																					\
	TSGDataType t dtype; write_header(&t);						                    \
																					\
	if (fwrite(&num_feat, sizeof(int32_t), 1, file)!=1 ||							\
			fwrite(&num_vec, sizeof(int32_t), 1, file)!=1 ||						\
			fwrite(matrix, sizeof(sg_type)*num_feat, num_vec, file)!=(size_t) num_vec)	\
		SG_ERROR("Failed to write Matrix\n");										\
}
SET_MATRIX(set_char_matrix, char, (CT_MATRIX, ST_NONE, PT_CHAR))
SET_MATRIX(set_byte_matrix, uint8_t, (CT_MATRIX, ST_NONE, PT_UINT8))
SET_MATRIX(set_int8_matrix, int8_t, (CT_MATRIX, ST_NONE, PT_INT8))
SET_MATRIX(set_int_matrix, int32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_MATRIX(set_uint_matrix, uint32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_MATRIX(set_long_matrix, int64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_MATRIX(set_ulong_matrix, uint64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_MATRIX(set_short_matrix, int16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_MATRIX(set_word_matrix, uint16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_MATRIX(set_shortreal_matrix, float32_t, (CT_MATRIX, ST_NONE, PT_FLOAT32))
SET_MATRIX(set_real_matrix, float64_t, (CT_MATRIX, ST_NONE, PT_FLOAT64))
SET_MATRIX(set_longreal_matrix, floatmax_t, (CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef SET_MATRIX

#define SET_NDARRAY(fname,sg_type,datatype)									\
void CBinaryFile::fname(const sg_type * array, int32_t * dims,int32_t num_dims)	\
{																			\
	size_t total = 1;														\
																			\
	if (!file)																\
		SG_ERROR("File invalid.\n");										\
																			\
	if (!array)																\
		SG_ERROR("Invalid array!\n");										\
																			\
	TSGDataType t datatype;													\
	write_header(&t);														\
																			\
	if (fwrite(&num_dims,sizeof(int32_t),1,file) != 1)						\
		SG_ERROR("Failed to write number of dimensions!\n");				\
																			\
	if (fwrite(dims,sizeof(int32_t),num_dims,file) != (size_t)num_dims)		\
		SG_ERROR("Failed to write sizes of dimensions!\n");					\
																			\
	for (int32_t i = 0;i < num_dims;i++)										\
		total *= dims[i];													\
																			\
	if (fwrite(array,sizeof(sg_type),total,file) != (size_t)total)			\
		SG_ERROR("Failed to write array data!\n");							\
}

SET_NDARRAY(set_byte_ndarray,uint8_t,(CT_NDARRAY, ST_NONE, PT_UINT8));
SET_NDARRAY(set_char_ndarray,char,(CT_NDARRAY, ST_NONE, PT_CHAR));
SET_NDARRAY(set_int_ndarray,int32_t,(CT_NDARRAY, ST_NONE, PT_INT32));
SET_NDARRAY(set_short_ndarray,int16_t,(CT_NDARRAY, ST_NONE, PT_INT16));
SET_NDARRAY(set_word_ndarray,uint16_t,(CT_NDARRAY, ST_NONE, PT_UINT16));
SET_NDARRAY(set_shortreal_ndarray,float32_t,(CT_NDARRAY, ST_NONE, PT_FLOAT32));
SET_NDARRAY(set_real_ndarray,float64_t,(CT_NDARRAY, ST_NONE, PT_FLOAT64));
#undef SET_NDARRAY

#define SET_SPARSEMATRIX(fname, sg_type, dtype) 			\
void CBinaryFile::fname(const SGSparseMatrix<sg_type>* matrix, 	\
		int32_t num_feat, int32_t num_vec)					\
{															\
	if (!(file && matrix))									\
		SG_ERROR("File or matrix invalid.\n");				\
															\
	TSGDataType t dtype; write_header(&t);					\
															\
	if (fwrite(&num_vec, sizeof(int32_t), 1, file)!=1)		\
		SG_ERROR("Failed to write Sparse Matrix\n");		\
															\
	for (int32_t i=0; i<num_vec; i++)						\
	{														\
		SGSparseMatrixEntry<sg_type>* vec = matrix[i].features;	\
		int32_t len=matrix[i].num_feat_entries;				\
		if ((fwrite(&len, sizeof(int32_t), 1, file)!=1) ||	\
				(fwrite(vec, sizeof(SGSparseMatrixEntry<sg_type>), len, file)!= (size_t) len))		\
			SG_ERROR("Failed to write Sparse Matrix\n");	\
	}														\
}
SET_SPARSEMATRIX(set_bool_sparsematrix, bool, (CT_MATRIX, ST_NONE, PT_BOOL))
SET_SPARSEMATRIX(set_char_sparsematrix, char, (CT_MATRIX, ST_NONE, PT_CHAR))
SET_SPARSEMATRIX(set_byte_sparsematrix, uint8_t, (CT_MATRIX, ST_NONE, PT_UINT8))
SET_SPARSEMATRIX(set_int8_sparsematrix, int8_t, (CT_MATRIX, ST_NONE, PT_INT8))
SET_SPARSEMATRIX(set_int_sparsematrix, int32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_SPARSEMATRIX(set_uint_sparsematrix, uint32_t, (CT_MATRIX, ST_NONE, PT_INT32))
SET_SPARSEMATRIX(set_long_sparsematrix, int64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_SPARSEMATRIX(set_ulong_sparsematrix, uint64_t, (CT_MATRIX, ST_NONE, PT_INT64))
SET_SPARSEMATRIX(set_short_sparsematrix, int16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_SPARSEMATRIX(set_word_sparsematrix, uint16_t, (CT_MATRIX, ST_NONE, PT_INT16))
SET_SPARSEMATRIX(set_shortreal_sparsematrix, float32_t, (CT_MATRIX, ST_NONE, PT_FLOAT32))
SET_SPARSEMATRIX(set_real_sparsematrix, float64_t, (CT_MATRIX, ST_NONE, PT_FLOAT64))
SET_SPARSEMATRIX(set_longreal_sparsematrix, floatmax_t, (CT_MATRIX, ST_NONE, PT_FLOATMAX))
#undef SET_SPARSEMATRIX

#define SET_STRING_LIST(fname, sg_type, dtype) \
void CBinaryFile::fname(const SGString<sg_type>* strings, int32_t num_str)	\
{																						\
	if (!(file && strings))																\
		SG_ERROR("File or strings invalid.\n");											\
																						\
	TSGDataType t dtype; write_header(&t);								                \
	for (int32_t i=0; i<num_str; i++)													\
	{																					\
		int32_t len = strings[i].length;												\
		if ((fwrite(&len, sizeof(int32_t), 1, file)!=1) ||								\
				(fwrite(strings[i].string, sizeof(sg_type), len, file)!= (size_t) len))	\
			SG_ERROR("Failed to write Sparse Matrix\n");								\
	}																					\
}
SET_STRING_LIST(set_char_string_list, char, (CT_VECTOR, ST_NONE, PT_CHAR))
SET_STRING_LIST(set_byte_string_list, uint8_t, (CT_VECTOR, ST_NONE, PT_UINT8))
SET_STRING_LIST(set_int8_string_list, int8_t, (CT_VECTOR, ST_NONE, PT_INT8))
SET_STRING_LIST(set_int_string_list, int32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_STRING_LIST(set_uint_string_list, uint32_t, (CT_VECTOR, ST_NONE, PT_INT32))
SET_STRING_LIST(set_long_string_list, int64_t, (CT_VECTOR, ST_NONE, PT_INT64))
SET_STRING_LIST(set_ulong_string_list, uint64_t, (CT_VECTOR, ST_NONE, PT_INT64))
SET_STRING_LIST(set_short_string_list, int16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_STRING_LIST(set_word_string_list, uint16_t, (CT_VECTOR, ST_NONE, PT_INT16))
SET_STRING_LIST(set_shortreal_string_list, float32_t, (CT_VECTOR, ST_NONE, PT_FLOAT32))
SET_STRING_LIST(set_real_string_list, float64_t, (CT_VECTOR, ST_NONE, PT_FLOAT64))
SET_STRING_LIST(set_longreal_string_list, floatmax_t, (CT_VECTOR, ST_NONE, PT_FLOATMAX))
#undef SET_STRING_LIST


int32_t CBinaryFile::parse_first_header(TSGDataType& type)
{
	    return -1;
}

int32_t CBinaryFile::parse_next_header(TSGDataType& type)
{
	    return -1;
}

void
CBinaryFile::read_header(TSGDataType* dest)
{
	ASSERT(file);

	char fourcc[4];
	uint16_t endian=0;

	if (!((fread(&fourcc, sizeof(char), 4, file)==4) &&
			(fread(&endian, sizeof(uint16_t), 1, file)== 1) &&
			(fread(&dest->m_ctype, sizeof(dest->m_ctype), 1, file)== 1)
		  && (fread(&dest->m_ptype, sizeof(dest->m_ptype), 1, file)== 1)
			))
		SG_ERROR("Error reading header\n");

	if (strncmp(fourcc, "SG01", 4))
		SG_ERROR("Header mismatch, expected SG01\n");
}

void
CBinaryFile::write_header(const TSGDataType* datatype)
{
	ASSERT(file);

	const char* fourcc="SG01";
	uint16_t endian=0x1234;

	if (!((fwrite(fourcc, sizeof(char), 4, file)==4) &&
		  (fwrite(&endian, sizeof(uint16_t), 1, file)==1) &&
		  (fwrite(&datatype->m_ctype, sizeof(datatype->m_ctype), 1,
				  file)==1)
		  && (fwrite(&datatype->m_ptype, sizeof(datatype->m_ptype), 1,
					 file)==1)
			))
		SG_ERROR("Error writing header\n");
}
