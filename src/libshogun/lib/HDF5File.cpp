/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#include <shogun/lib/config.h>

#ifdef HAVE_HDF5
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <hdf5.h>

#include "lib/HDF5File.h"

#include "features/StringFeatures.h"
#include "features/SparseFeatures.h"

using namespace shogun;

CHDF5File::CHDF5File(char* fname, char rw, const char* name) : CFile()
{
	if (name)
		set_variable_name(name);

	switch (rw)
	{
		case 'r':
			h5file = H5Fopen(fname, H5F_ACC_RDONLY, H5P_DEFAULT);
			break;
		case 'w':
			h5file = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
			break;
		case 'a':
			h5file = H5Fopen(fname, H5F_ACC_RDWR, H5P_DEFAULT);
			break;
		default:
			SG_ERROR("unknown mode '%c'\n", rw);
	};
}

CHDF5File::~CHDF5File()
{
	H5Fclose(h5file);
}

//dataset = H5Dopen(file, "/Data/CData");
//status = H5Dclose(dataset);
//
//H5Gcreate(file, "/Data", 0);
//  herr_t status;
//    status = H5Gclose(group);
//
//
//grp = H5Gcreate2(file, "/Data", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
//

#define GET_VECTOR(fname, sg_type, datatype)										\
void CHDF5File::fname(sg_type*& vec, int32_t& len)									\
{																					\
	if (!h5file)																	\
		SG_ERROR("File invalid.\n");												\
	hid_t dataset = H5Dopen(h5file, variable_name);                                 \
	hid_t dtype  = H5Dget_type(dataset);                                            \
	hid_t dataspace = H5Dget_space(dataset);										\
																					\
	H5T_class_t t_class=H5Tget_class(dtype);                                        \
	H5T_sign_t sgn=H5Tget_sign(dtype);												\
			switch (sgn)                                                            \
			{                                                                       \
				case H5T_SGN_NONE:                                                  \
					printf("false");                                                \
					break;                                                          \
				case H5T_SGN_2:                                                     \
					printf("true");                                                 \
					break;                                                          \
				default:															\
					printf("unknown");                                              \
					break;															\
			}                                                                       \
	size_t sz=H5Tget_size(dtype);													\
			SG_PRINT("Size=%d\n", sz);                                              \
                                                                                    \
	switch (t_class)                                                                \
	{                                                                               \
		case H5T_INTEGER:                                                           \
			SG_PRINT("int\n");                                                    	\
			break;                                                                  \
		case H5T_FLOAT:                                                             \
			SG_PRINT("float\n");                                                    \
			break;                                                                  \
		case H5T_STRING:                                                            \
			SG_PRINT("string\n");                                                   \
			break;                                                                  \
		case H5T_VLEN:                                                              \
			SG_PRINT("vlen\n");                                                     \
			break;                                                                  \
		case H5T_ARRAY:                                                             \
			SG_PRINT("array\n");                                                    \
			break;                                                                  \
		default:																	\
			SG_ERROR("Datatype mismatch\n");										\
			break;																	\
	}                                                                               \
																					\
	len=H5Sget_simple_extent_npoints(dataspace);									\
	vec=new sg_type[len];															\
	herr_t status = H5Dread(dataset, H5T_NATIVE_DOUBLE, H5S_ALL, 					\
			H5S_ALL, H5P_DEFAULT, vec);												\
																					\
   status = H5Dclose(dataset);														\
}

	/*
	hsize_t     adims_out[2];
	 * array_rank_out = H5Tget_array_ndims(datatype);
	status = H5Tget_array_dims(datatype, adims_out, NULL); 
																					\
	if (fread(&len, sizeof(int32_t), 1, file)!=1)									\
		SG_ERROR("Failed to read vector length\n");									\
	vec=new sg_type[len];															\
	if (fread(vec, sizeof(sg_type), len, file)!=(size_t) len)						\
		SG_ERROR("Failed to read Matrix\n");									*/	\
//}
/*
H5T_NATIVE_CHAR	char
H5T_NATIVE_SCHAR	signed char
H5T_NATIVE_UCHAR	unsigned char
H5T_NATIVE_SHORT	short
H5T_NATIVE_USHORT	unsigned short
H5T_NATIVE_INT	int
H5T_NATIVE_UINT	unsigned
H5T_NATIVE_LONG	long
H5T_NATIVE_ULONG	unsigned long
H5T_NATIVE_LLONG	long long
H5T_NATIVE_ULLONG	unsigned long long
H5T_NATIVE_FLOAT	float
H5T_NATIVE_DOUBLE	double
H5T_NATIVE_LDOUBLE	long double
H5T_NATIVE_HSIZE	hsize_t
H5T_NATIVE_HSSIZE	hssize_t
H5T_NATIVE_HERR	herr_t
H5T_NATIVE_HBOOL	hbool_t
*/

GET_VECTOR(get_bool_vector, bool, DT_VECTOR_BOOL)
GET_VECTOR(get_byte_vector, uint8_t, DT_VECTOR_BYTE)
GET_VECTOR(get_char_vector, char, DT_VECTOR_CHAR)
GET_VECTOR(get_int_vector, int32_t, DT_VECTOR_INT)
GET_VECTOR(get_shortreal_vector, float32_t, DT_VECTOR_SHORTREAL)
GET_VECTOR(get_real_vector, float64_t, DT_VECTOR_REAL)
GET_VECTOR(get_short_vector, int16_t, DT_VECTOR_SHORT)
GET_VECTOR(get_word_vector, uint16_t, DT_VECTOR_WORD)
#undef GET_VECTOR

#define GET_MATRIX(fname, sg_type, datatype)										\
void CHDF5File::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)		\
{																					\
}

GET_MATRIX(get_bool_matrix, bool, DT_DENSE_BOOL)
GET_MATRIX(get_char_matrix, char, DT_DENSE_CHAR)
GET_MATRIX(get_byte_matrix, uint8_t, DT_DENSE_BYTE)
GET_MATRIX(get_int_matrix, int32_t, DT_DENSE_INT)
GET_MATRIX(get_uint_matrix, uint32_t, DT_DENSE_UINT)
GET_MATRIX(get_long_matrix, int64_t, DT_DENSE_LONG)
GET_MATRIX(get_ulong_matrix, uint64_t, DT_DENSE_ULONG)
GET_MATRIX(get_short_matrix, int16_t, DT_DENSE_SHORT)
GET_MATRIX(get_word_matrix, uint16_t, DT_DENSE_WORD)
GET_MATRIX(get_shortreal_matrix, float32_t, DT_DENSE_SHORTREAL)
GET_MATRIX(get_real_matrix, float64_t, DT_DENSE_REAL)
GET_MATRIX(get_longreal_matrix, floatmax_t, DT_DENSE_LONGREAL)
#undef GET_MATRIX

void CHDF5File::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CHDF5File::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

#define GET_SPARSEMATRIX(fname, sg_type, datatype)										\
void CHDF5File::fname(TSparse<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																						\
	if (!(file))																		\
		SG_ERROR("File invalid.\n");													\
}
GET_SPARSEMATRIX(get_bool_sparsematrix, bool, DT_SPARSE_BOOL)
GET_SPARSEMATRIX(get_char_sparsematrix, char, DT_SPARSE_CHAR)
GET_SPARSEMATRIX(get_byte_sparsematrix, uint8_t, DT_SPARSE_BYTE)
GET_SPARSEMATRIX(get_int_sparsematrix, int32_t, DT_SPARSE_INT)
GET_SPARSEMATRIX(get_uint_sparsematrix, uint32_t, DT_SPARSE_UINT)
GET_SPARSEMATRIX(get_long_sparsematrix, int64_t, DT_SPARSE_LONG)
GET_SPARSEMATRIX(get_ulong_sparsematrix, uint64_t, DT_SPARSE_ULONG)
GET_SPARSEMATRIX(get_short_sparsematrix, int16_t, DT_SPARSE_SHORT)
GET_SPARSEMATRIX(get_word_sparsematrix, uint16_t, DT_SPARSE_WORD)
GET_SPARSEMATRIX(get_shortreal_sparsematrix, float32_t, DT_SPARSE_SHORTREAL)
GET_SPARSEMATRIX(get_real_sparsematrix, float64_t, DT_SPARSE_REAL)
GET_SPARSEMATRIX(get_longreal_sparsematrix, floatmax_t, DT_SPARSE_LONGREAL)
#undef GET_SPARSEMATRIX


#define GET_STRING_LIST(fname, sg_type, datatype)												\
void CHDF5File::fname(T_STRING<sg_type>*& strings, int32_t& num_str, int32_t& max_string_len) \
{																								\
}

GET_STRING_LIST(get_bool_string_list, bool, DT_STRING_BOOL)
GET_STRING_LIST(get_char_string_list, char, DT_STRING_CHAR)
GET_STRING_LIST(get_byte_string_list, uint8_t, DT_STRING_BYTE)
GET_STRING_LIST(get_int_string_list, int32_t, DT_STRING_INT)
GET_STRING_LIST(get_uint_string_list, uint32_t, DT_STRING_UINT)
GET_STRING_LIST(get_long_string_list, int64_t, DT_STRING_LONG)
GET_STRING_LIST(get_ulong_string_list, uint64_t, DT_STRING_ULONG)
GET_STRING_LIST(get_short_string_list, int16_t, DT_STRING_SHORT)
GET_STRING_LIST(get_word_string_list, uint16_t, DT_STRING_WORD)
GET_STRING_LIST(get_shortreal_string_list, float32_t, DT_STRING_SHORTREAL)
GET_STRING_LIST(get_real_string_list, float64_t, DT_STRING_REAL)
GET_STRING_LIST(get_longreal_string_list, floatmax_t, DT_STRING_LONGREAL)
#undef GET_STRING_LIST

/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, sg_type, dtype)							\
void CHDF5File::fname(const sg_type* vec, int32_t len)			\
{																	\
	if (!(file && vec))												\
		SG_ERROR("File or vector invalid.\n");						\
																	\
	/*write_header(dtype);											\
																	\
	if (fwrite(&len, sizeof(int32_t), 1, file)!=1 ||				\
			fwrite(vec, sizeof(sg_type), len, file)!=(size_t) len)	\
		SG_ERROR("Failed to write vector\n");					*/	\
}
SET_VECTOR(set_bool_vector, bool, DT_VECTOR_BOOL)
SET_VECTOR(set_byte_vector, uint8_t, DT_VECTOR_BYTE)
SET_VECTOR(set_char_vector, char, DT_VECTOR_CHAR)
SET_VECTOR(set_int_vector, int32_t, DT_VECTOR_INT)
SET_VECTOR(set_shortreal_vector, float32_t, DT_VECTOR_SHORTREAL)
SET_VECTOR(set_real_vector, float64_t, DT_VECTOR_REAL)
SET_VECTOR(set_short_vector, int16_t, DT_VECTOR_SHORT)
SET_VECTOR(set_word_vector, uint16_t, DT_VECTOR_WORD)
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, dtype) \
void CHDF5File::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																					\
	if (!(file && matrix))															\
		SG_ERROR("File or matrix invalid.\n");										\
																					\
}
SET_MATRIX(set_bool_matrix, bool, DT_DENSE_BOOL)
SET_MATRIX(set_char_matrix, char, DT_DENSE_CHAR)
SET_MATRIX(set_byte_matrix, uint8_t, DT_DENSE_BYTE)
SET_MATRIX(set_int_matrix, int32_t, DT_DENSE_INT)
SET_MATRIX(set_uint_matrix, uint32_t, DT_DENSE_UINT)
SET_MATRIX(set_long_matrix, int64_t, DT_DENSE_LONG)
SET_MATRIX(set_ulong_matrix, uint64_t, DT_DENSE_ULONG)
SET_MATRIX(set_short_matrix, int16_t, DT_DENSE_SHORT)
SET_MATRIX(set_word_matrix, uint16_t, DT_DENSE_WORD)
SET_MATRIX(set_shortreal_matrix, float32_t, DT_DENSE_SHORTREAL)
SET_MATRIX(set_real_matrix, float64_t, DT_DENSE_REAL)
SET_MATRIX(set_longreal_matrix, floatmax_t, DT_DENSE_LONGREAL)
#undef SET_MATRIX

#define SET_SPARSEMATRIX(fname, sg_type, dtype) 			\
void CHDF5File::fname(const TSparse<sg_type>* matrix, 	\
		int32_t num_feat, int32_t num_vec)					\
{															\
	if (!(file && matrix))									\
		SG_ERROR("File or matrix invalid.\n");				\
															\
}
SET_SPARSEMATRIX(set_bool_sparsematrix, bool, DT_SPARSE_BOOL)
SET_SPARSEMATRIX(set_char_sparsematrix, char, DT_SPARSE_CHAR)
SET_SPARSEMATRIX(set_byte_sparsematrix, uint8_t, DT_SPARSE_BYTE)
SET_SPARSEMATRIX(set_int_sparsematrix, int32_t, DT_SPARSE_INT)
SET_SPARSEMATRIX(set_uint_sparsematrix, uint32_t, DT_SPARSE_UINT)
SET_SPARSEMATRIX(set_long_sparsematrix, int64_t, DT_SPARSE_LONG)
SET_SPARSEMATRIX(set_ulong_sparsematrix, uint64_t, DT_SPARSE_ULONG)
SET_SPARSEMATRIX(set_short_sparsematrix, int16_t, DT_SPARSE_SHORT)
SET_SPARSEMATRIX(set_word_sparsematrix, uint16_t, DT_SPARSE_WORD)
SET_SPARSEMATRIX(set_shortreal_sparsematrix, float32_t, DT_SPARSE_SHORTREAL)
SET_SPARSEMATRIX(set_real_sparsematrix, float64_t, DT_SPARSE_REAL)
SET_SPARSEMATRIX(set_longreal_sparsematrix, floatmax_t, DT_SPARSE_LONGREAL)
#undef SET_SPARSEMATRIX

#define SET_STRING_LIST(fname, sg_type, dtype) \
void CHDF5File::fname(const T_STRING<sg_type>* strings, int32_t num_str)	\
{																						\
	if (!(file && strings))																\
		SG_ERROR("File or strings invalid.\n");											\
																						\
}
SET_STRING_LIST(set_bool_string_list, bool, DT_STRING_BOOL)
SET_STRING_LIST(set_char_string_list, char, DT_STRING_CHAR)
SET_STRING_LIST(set_byte_string_list, uint8_t, DT_STRING_BYTE)
SET_STRING_LIST(set_int_string_list, int32_t, DT_STRING_INT)
SET_STRING_LIST(set_uint_string_list, uint32_t, DT_STRING_UINT)
SET_STRING_LIST(set_long_string_list, int64_t, DT_STRING_LONG)
SET_STRING_LIST(set_ulong_string_list, uint64_t, DT_STRING_ULONG)
SET_STRING_LIST(set_short_string_list, int16_t, DT_STRING_SHORT)
SET_STRING_LIST(set_word_string_list, uint16_t, DT_STRING_WORD)
SET_STRING_LIST(set_shortreal_string_list, float32_t, DT_STRING_SHORTREAL)
SET_STRING_LIST(set_real_string_list, float64_t, DT_STRING_REAL)
SET_STRING_LIST(set_longreal_string_list, floatmax_t, DT_STRING_LONGREAL)
#undef SET_STRING_LIST

#endif //  HDF5
