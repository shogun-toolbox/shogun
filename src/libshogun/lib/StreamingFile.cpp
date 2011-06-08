/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */

#include "features/SparseFeatures.h"
#include "lib/File.h"
#include "lib/StreamingFile.h"
#include "lib/AsciiFile.h"
#include "lib/Mathematics.h"
#include <ctype.h>

using namespace shogun;

CStreamingFile::CStreamingFile(void)
{
	SG_UNSTABLE("CStreamingFile::CStreamingFile(void)", "\n");
}

CStreamingFile::CStreamingFile(FILE* f, const char* name) : CFile(f, name)
{
}

CStreamingFile::CStreamingFile(char* fname, char rw, const char* name) : CFile(fname, rw, name)
{
}

CStreamingFile::~CStreamingFile()
{
}


#define GET_VECTOR(fname, conv, sg_type)						\
void CStreamingFile::fname(sg_type*& vector, int32_t& num_feat)	\
{																\
	size_t buffer_size=1024;									\
	char* buffer=new char[buffer_size];							\
	ssize_t bytes_read;											\
																\
	bytes_read=CAsciiFile::getline(&buffer, &buffer_size, file);	\
																\
																\
	if (bytes_read<=0)											\
	{															\
		vector=NULL;											\
		num_feat=-1;											\
		return;													\
	}															\
																\
																\
	SG_DEBUG("line read from file:\n%s\n", buffer);				\
																\
	/* determine num_feat, populate dynamic array */			\
	int32_t nf=0;												\
	num_feat=0;													\
																\
	char* ptr_item=NULL;										\
	char* ptr_data=buffer;										\
	DynArray<char*>* items=new DynArray<char*>();				\
																\
	while (*ptr_data)											\
	{															\
		if ((*ptr_data=='\n') ||								\
			(ptr_data - buffer >= bytes_read - 1))				\
		{														\
			if (ptr_item)										\
				nf++;											\
																\
			append_item(items, ptr_data, ptr_item);				\
			num_feat=nf;										\
																\
			nf=0;												\
			ptr_item=NULL;										\
			break;												\
		}														\
		else if (!isblank(*ptr_data) && !ptr_item)				\
		{														\
			ptr_item=ptr_data;									\
		}														\
		else if (isblank(*ptr_data) && ptr_item)				\
		{														\
			append_item(items, ptr_data, ptr_item);				\
			ptr_item=NULL;										\
			nf++;												\
		}														\
																\
		ptr_data++;												\
	}															\
																\
	SG_DEBUG("num_feat %d\n", num_feat);						\
	delete buffer;												\
																\
	/* now copy data into vector */								\
	vector=new sg_type[num_feat];								\
	for (int32_t i=0; i<num_feat; i++)							\
	{															\
		char* item=items->get_element(i);						\
		vector[i]=conv(item);									\
		delete[] item;											\
	}															\
	delete items;												\
}

GET_VECTOR(get_byte_vector, atoi, uint8_t)
GET_VECTOR(get_char_vector, atoi, char)
GET_VECTOR(get_int_vector, atoi, int32_t)
GET_VECTOR(get_shortreal_vector, atof, float32_t)
GET_VECTOR(get_real_vector, atof, float64_t)
GET_VECTOR(get_short_vector, atoi, int16_t)
GET_VECTOR(get_word_vector, atoi, uint16_t)
#undef GET_VECTOR

#define GET_MATRIX(fname, conv, sg_type)								\
void CStreamingFile::fname(sg_type*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{																		\
}

GET_MATRIX(get_byte_matrix, atoi, uint8_t)
GET_MATRIX(get_int8_matrix, atoi, int8_t)
GET_MATRIX(get_char_matrix, atoi, char)
GET_MATRIX(get_int_matrix, atoi, int32_t)
GET_MATRIX(get_uint_matrix, atoi, uint32_t)
GET_MATRIX(get_long_matrix, atoll, int64_t)
GET_MATRIX(get_ulong_matrix, atoll, uint64_t)
GET_MATRIX(get_shortreal_matrix, atof, float32_t)
GET_MATRIX(get_real_matrix, atof, float64_t)
GET_MATRIX(get_longreal_matrix, atof, floatmax_t)
GET_MATRIX(get_short_matrix, atoi, int16_t)
GET_MATRIX(get_word_matrix, atoi, uint16_t)
#undef GET_MATRIX


void CStreamingFile::get_byte_ndarray(uint8_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_char_ndarray(char*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_int_ndarray(int32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_shortreal_ndarray(float32_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_real_ndarray(float64_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_short_ndarray(int16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

void CStreamingFile::get_word_ndarray(uint16_t*& array, int32_t*& dims, int32_t& num_dims)
{
}

#define GET_SPARSEMATRIX(fname, conv, sg_type)										\
void CStreamingFile::fname(SGSparseVector<sg_type>*& matrix, int32_t& num_feat, int32_t& num_vec)	\
{	\
}

GET_SPARSEMATRIX(get_bool_sparsematrix, atoi, bool)
GET_SPARSEMATRIX(get_byte_sparsematrix, atoi, uint8_t)
GET_SPARSEMATRIX(get_int8_sparsematrix, atoi, int8_t)
GET_SPARSEMATRIX(get_char_sparsematrix, atoi, char)
GET_SPARSEMATRIX(get_int_sparsematrix, atoi, int32_t)
GET_SPARSEMATRIX(get_uint_sparsematrix, atoi, uint32_t)
GET_SPARSEMATRIX(get_long_sparsematrix, atoll, int64_t)
GET_SPARSEMATRIX(get_ulong_sparsematrix, atoll, uint64_t)
GET_SPARSEMATRIX(get_shortreal_sparsematrix, atof, float32_t)
GET_SPARSEMATRIX(get_real_sparsematrix, atof, float64_t)
GET_SPARSEMATRIX(get_longreal_sparsematrix, atof, floatmax_t)
GET_SPARSEMATRIX(get_short_sparsematrix, atoi, int16_t)
GET_SPARSEMATRIX(get_word_sparsematrix, atoi, uint16_t)
#undef GET_SPARSEMATRIX


void CStreamingFile::get_byte_string_list(SGString<uint8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CStreamingFile::get_int8_string_list(SGString<int8_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CStreamingFile::get_char_string_list(SGString<char>*& strings, int32_t& num_str, int32_t& max_string_len)
{
}

void CStreamingFile::get_int_string_list(SGString<int32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_uint_string_list(SGString<uint32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_short_string_list(SGString<int16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_word_string_list(SGString<uint16_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_long_string_list(SGString<int64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_ulong_string_list(SGString<uint64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_shortreal_string_list(SGString<float32_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_real_string_list(SGString<float64_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}

void CStreamingFile::get_longreal_string_list(SGString<floatmax_t>*& strings, int32_t& num_str, int32_t& max_string_len)
{
	strings=NULL;
	num_str=0;
	max_string_len=0;
}


/** set functions - to pass data from shogun to the target interface */

#define SET_VECTOR(fname, mfname, sg_type)	\
void CStreamingFile::fname(const sg_type* vec, int32_t len)	\
{															\
	mfname(vec, len, 1);									\
}
SET_VECTOR(set_byte_vector, set_byte_matrix, uint8_t)
SET_VECTOR(set_char_vector, set_char_matrix, char)
SET_VECTOR(set_int_vector, set_int_matrix, int32_t)
SET_VECTOR(set_shortreal_vector, set_shortreal_matrix, float32_t)
SET_VECTOR(set_real_vector, set_real_matrix, float64_t)
SET_VECTOR(set_short_vector, set_short_matrix, int16_t)
SET_VECTOR(set_word_vector, set_word_matrix, uint16_t)
#undef SET_VECTOR

#define SET_MATRIX(fname, sg_type, fprt_type, type_str) \
void CStreamingFile::fname(const sg_type* matrix, int32_t num_feat, int32_t num_vec)	\
{																		\
}
SET_MATRIX(set_char_matrix, char, char, "%c")
SET_MATRIX(set_byte_matrix, uint8_t, uint8_t, "%u")
SET_MATRIX(set_int8_matrix, int8_t, int8_t, "%d")
SET_MATRIX(set_int_matrix, int32_t, int32_t, "%i")
SET_MATRIX(set_uint_matrix, uint32_t, uint32_t, "%u")
SET_MATRIX(set_long_matrix, int64_t, long long int, "%lli")
SET_MATRIX(set_ulong_matrix, uint64_t, long long unsigned int, "%llu")
SET_MATRIX(set_short_matrix, int16_t, int16_t, "%i")
SET_MATRIX(set_word_matrix, uint16_t, uint16_t, "%u")
SET_MATRIX(set_shortreal_matrix, float32_t, float32_t, "%f")
SET_MATRIX(set_real_matrix, float64_t, float64_t, "%f")
SET_MATRIX(set_longreal_matrix, floatmax_t, floatmax_t, "%Lf")
#undef SET_MATRIX

#define SET_SPARSEMATRIX(fname, sg_type, fprt_type, type_str) \
void CStreamingFile::fname(const SGSparseVector<sg_type>* matrix, int32_t num_feat, int32_t num_vec)	\
{																		\
}

SET_SPARSEMATRIX(set_bool_sparsematrix, bool, uint8_t, "%u")
SET_SPARSEMATRIX(set_char_sparsematrix, char, char, "%c")
SET_SPARSEMATRIX(set_byte_sparsematrix, uint8_t, uint8_t, "%u")
SET_SPARSEMATRIX(set_int8_sparsematrix, int8_t, int8_t, "%d")
SET_SPARSEMATRIX(set_int_sparsematrix, int32_t, int32_t, "%i")
SET_SPARSEMATRIX(set_uint_sparsematrix, uint32_t, uint32_t, "%u")
SET_SPARSEMATRIX(set_long_sparsematrix, int64_t, long long int, "%lli")
SET_SPARSEMATRIX(set_ulong_sparsematrix, uint64_t, long long unsigned int, "%llu")
SET_SPARSEMATRIX(set_short_sparsematrix, int16_t, int16_t, "%i")
SET_SPARSEMATRIX(set_word_sparsematrix, uint16_t, uint16_t, "%u")
SET_SPARSEMATRIX(set_shortreal_sparsematrix, float32_t, float32_t, "%f")
SET_SPARSEMATRIX(set_real_sparsematrix, float64_t, float64_t, "%f")
SET_SPARSEMATRIX(set_longreal_sparsematrix, floatmax_t, floatmax_t, "%Lf")
#undef SET_SPARSEMATRIX

void CStreamingFile::set_byte_string_list(const SGString<uint8_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_int8_string_list(const SGString<int8_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_char_string_list(const SGString<char>* strings, int32_t num_str)
{
}

void CStreamingFile::set_int_string_list(const SGString<int32_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_uint_string_list(const SGString<uint32_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_short_string_list(const SGString<int16_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_word_string_list(const SGString<uint16_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_long_string_list(const SGString<int64_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_ulong_string_list(const SGString<uint64_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_shortreal_string_list(const SGString<float32_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_real_string_list(const SGString<float64_t>* strings, int32_t num_str)
{
}

void CStreamingFile::set_longreal_string_list(const SGString<floatmax_t>* strings, int32_t num_str)
{
}


template <class T> void CStreamingFile::append_item(
	DynArray<T>* items, char* ptr_data, char* ptr_item)
{
	size_t len=(ptr_data-ptr_item)/sizeof(char);
	char* item=new char[len+1];
	memset(item, 0, sizeof(char)*(len+1));
	item=strncpy(item, ptr_item, len);

	SG_DEBUG("current %c, len %d, item %s\n", *ptr_data, len, item);
	items->append_element(item);
}
