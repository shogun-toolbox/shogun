/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include "base/SGObject.h"

namespace shogun
{
template <class ST> struct T_STRING;
template <class ST> struct TSparse;

/* Datatypes that shogun supports. */
/*
enum DataType
{
	UNDEFINED,

	///simple scalar/string types
	SCALAR_INT,
	SCALAR_REAL,
	SCALAR_BOOL,
	STANDARD_STRING,

	///vector type
	VECTOR_BOOL,
	VECTOR_BYTE,
	VECTOR_CHAR,
	VECTOR_INT,
	VECTOR_REAL,
	VECTOR_SHORTREAL,
	VECTOR_SHORT,
	VECTOR_WORD,

	///dense matrices 
	DENSE_INT,
	DENSE_REAL,
	DENSE_SHORTREAL,
	DENSE_SHORT,
	DENSE_WORD,

	///dense nd arrays
	NDARRAY_BYTE,
	NDARRAY_CHAR,
	NDARRAY_INT,
	NDARRAY_REAL,
	NDARRAY_SHORTREAL,
	NDARRAY_SHORT,
	NDARRAY_WORD,

	///sparse matrices
	SPARSE_BYTE,
	SPARSE_CHAR,
	SPARSE_INT,
	SPARSE_REAL,
	SPARSE_SHORT,
	SPARSE_SHORTREAL,
	SPARSE_WORD,

	///strings of arbitrary type
	STRING_BYTE,
	STRING_CHAR,
	STRING_INT,
	STRING_SHORT,
	STRING_WORD,

	/// structures
	ATTR_STRUCT
};*/

/** @brief A File access base class.
 *
 * A file is assumed to be a seekable raw data stream.
 *
 */
class CFile : public CSGObject
{
public:
	/** constructor
	 *
	 * @param f already opened file
	 */
	CFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CFile(char* fname, char rw='r', const char* name=NULL);

	virtual ~CFile();

	void set_variable_name(const char* name);
	char* get_variable_name();

	/** get data type of current element */
	/*virtual DataType get_data_type()=0;*/

	/** vector access functions */
	/*virtual void get_vector(void*& vector, int32_t& len, DataType& dtype);*/

	virtual void get_bool_vector(bool*& vector, int32_t& len);
	virtual void get_byte_vector(uint8_t*& vector, int32_t& len)=0;
	virtual void get_char_vector(char*& vector, int32_t& len)=0;
	virtual void get_int_vector(int32_t*& vector, int32_t& len)=0;
	virtual void get_real_vector(float64_t*& vector, int32_t& len)=0;
	virtual void get_shortreal_vector(float32_t*& vector, int32_t& len)=0;
	virtual void get_short_vector(int16_t*& vector, int32_t& len)=0;
	virtual void get_word_vector(uint16_t*& vector, int32_t& len)=0;

	/** matrix access functions */
	/*virtual void get_matrix(
			void*& matrix, int32_t& num_feat, int32_t& num_vec, DataType& dtype);*/

	virtual void get_bool_matrix(
			bool*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_byte_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_char_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_int_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_uint_matrix(
			uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_long_matrix(
			int64_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_ulong_matrix(
			uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_shortreal_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_real_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_longreal_matrix(
			floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_short_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_word_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;

	/** nd-array access functions */
	/*virtual void get_ndarray(
			void*& array, int32_t*& dims, int32_t& num_dims, DataType& dtype);*/

	virtual void get_byte_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_char_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_int_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_shortreal_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_real_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_short_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims)=0;
	virtual void get_word_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims)=0;

	virtual void get_real_sparsematrix(
			TSparse<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;

	/*  future versions might support types other than float64_t

		virtual void get_byte_sparsematrix(TSparse<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_char_sparsematrix(TSparse<char>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_int_sparsematrix(TSparse<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_shortreal_sparsematrix(TSparse<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_short_sparsematrix(TSparse<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
		virtual void get_word_sparsematrix(TSparse<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0; */

	virtual void get_bool_string_list(
			T_STRING<bool>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_byte_string_list(
			T_STRING<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_char_string_list(
			T_STRING<char>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_int_string_list(
			T_STRING<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_uint_string_list(
			T_STRING<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_short_string_list(
			T_STRING<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_word_string_list(
			T_STRING<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_long_string_list(
			T_STRING<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_ulong_string_list(
			T_STRING<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_shortreal_string_list(
			T_STRING<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_real_string_list(
			T_STRING<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_longreal_string_list(
			T_STRING<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;

	virtual void set_bool_vector(const bool* vector, int32_t len);
	virtual void set_byte_vector(const uint8_t* vector, int32_t len)=0;
	virtual void set_char_vector(const char* vector, int32_t len)=0;
	virtual void set_int_vector(const int32_t* vector, int32_t len)=0;
	virtual void set_shortreal_vector(
			const float32_t* vector, int32_t len)=0;
	virtual void set_real_vector(const float64_t* vector, int32_t len)=0;
	virtual void set_short_vector(const int16_t* vector, int32_t len)=0;
	virtual void set_word_vector(const uint16_t* vector, int32_t len)=0;


	virtual void set_bool_matrix(
			const bool* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_byte_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_char_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_int_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_uint_matrix(
			const uint32_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_long_matrix(
			const int64_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_ulong_matrix(
			const uint64_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_shortreal_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_real_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_longreal_matrix(
			const floatmax_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_short_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_word_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec)=0;

	virtual void set_real_sparsematrix(
			const TSparse<float64_t>* matrix, int32_t num_feat,
			int32_t num_vec, int64_t nnz)=0;

	/*  future versions might support types other than float64_t

		virtual void set_byte_sparsematrix(const TSparse<uint8_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_char_sparsematrix(const TSparse<char>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_int_sparsematrix(const TSparse<int32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_shortreal_sparsematrix(const TSparse<float32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_short_sparsematrix(const TSparse<int16_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
		virtual void set_word_sparsematrix(const TSparse<uint16_t>* matrix, int32_t num_feat, int32_t num_vec)=0; */


	virtual void set_bool_string_list(
			const T_STRING<bool>* strings, int32_t num_str);
	virtual void set_byte_string_list(
			const T_STRING<uint8_t>* strings, int32_t num_str)=0;
	virtual void set_char_string_list(
			const T_STRING<char>* strings, int32_t num_str)=0;
	virtual void set_int_string_list(
			const T_STRING<int32_t>* strings, int32_t num_str)=0;
	virtual void set_uint_string_list(
			const T_STRING<uint32_t>* strings, int32_t num_str)=0;
	virtual void set_short_string_list(
			const T_STRING<int16_t>* strings, int32_t num_str)=0;
	virtual void set_word_string_list(
			const T_STRING<uint16_t>* strings, int32_t num_str)=0;
	virtual void set_long_string_list(
			const T_STRING<int64_t>* strings, int32_t num_str)=0;
	virtual void set_ulong_string_list(
			const T_STRING<uint64_t>* strings, int32_t num_str)=0;
	virtual void set_shortreal_string_list(
			const T_STRING<float32_t>* strings, int32_t num_str)=0;
	virtual void set_real_string_list(
			const T_STRING<float64_t>* strings, int32_t num_str)=0;
	virtual void set_longreal_string_list(
			const T_STRING<floatmax_t>* strings, int32_t num_str)=0;

	/** @return object name */
	inline virtual const char* get_name() const { return "File"; }

protected:
	/** file object */
	FILE* file;
	/** task */
	char task;
	/** name of the handled file */
	char* filename;
	/** variable name / path to variable */
	char* variable_name;
};
}
#endif
