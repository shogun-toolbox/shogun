/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2009 Soeren Sonnenburg
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef __FILE_H__
#define __FILE_H__

#include <stdio.h>
#include "base/SGObject.h"
#include "lib/DataType.h"

namespace shogun
{
template <class ST> class SGString;
template <class ST> struct SGSparseMatrix;

/** @brief A File access base class.
 *
 * A file is assumed to be a seekable raw data stream.
 *
 * \sa CAsciiFile
 * \sa CBinaryFile
 * \sa CHDF5File
 *
 */
class CFile : public CSGObject
{
public:
	/** default constructor */
	CFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CFile(char* fname, char rw='r', const char* name=NULL);

	/** default destructor */
	virtual ~CFile();

	void close()
	{
		SG_FREE(variable_name);
		SG_FREE(filename);
		if (file)
		  fclose(file);
		variable_name=NULL;
		filename=NULL;
		file=NULL;
	}

	/** set the path to the variable to be accessed
	 *
	 * only supported by some file interfaces like CHDF5File
	 *
	 * @param name variable path & name
	 */
	void set_variable_name(const char* name);

	/** get the path to the variable to be accessed
	 *
	 * only supported by some file interfaces like CHDF5File
	 *
	 * @return name variable path & name
	 */
	char* get_variable_name();

	/** get data type of current element */
	/*virtual DataType get_data_type()=0;*/

	/** vector access functions */
	/*virtual void get_vector(void*& vector, int32_t& len, DataType& dtype);*/

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	virtual void get_bool_vector(bool*& vector, int32_t& len);
	virtual void get_byte_vector(uint8_t*& vector, int32_t& len)=0;
	virtual void get_char_vector(char*& vector, int32_t& len)=0;
	virtual void get_int_vector(int32_t*& vector, int32_t& len)=0;
	virtual void get_real_vector(float64_t*& vector, int32_t& len)=0;
	virtual void get_shortreal_vector(float32_t*& vector, int32_t& len)=0;
	virtual void get_short_vector(int16_t*& vector, int32_t& len)=0;
	virtual void get_word_vector(uint16_t*& vector, int32_t& len)=0;
	//@}

	/** matrix access functions */
	/*virtual void get_matrix(
			void*& matrix, int32_t& num_feat, int32_t& num_vec, DataType& dtype);*/

	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when loading matrices from e.g. file
	 * and return the matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_bool_matrix(
			bool*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_byte_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_int8_matrix(
			int8_t*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
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
	//@}

	/** nd-array access functions */
	/*virtual void get_ndarray(
			void*& array, int32_t*& dims, int32_t& num_dims, DataType& dtype);*/

	/** @name N-Dimensional Array Access Functions
	 *
	 * Functions to access n-dimensional arrays of one of the several base
	 * data types. These functions are used when loading n-dimensional arrays
	 * from e.g. file and return the them and its dimensions dims and num_dims
	 * by reference
	 */
	//@{
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
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_bool_sparsematrix(
			SGSparseMatrix<bool>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_byte_sparsematrix(
			SGSparseMatrix<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_int8_sparsematrix(
		SGSparseMatrix<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_char_sparsematrix(
			SGSparseMatrix<char>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_int_sparsematrix(
			SGSparseMatrix<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_uint_sparsematrix(
			SGSparseMatrix<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_long_sparsematrix(
			SGSparseMatrix<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_ulong_sparsematrix(
			SGSparseMatrix<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_short_sparsematrix(
			SGSparseMatrix<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_word_sparsematrix(
			SGSparseMatrix<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_shortreal_sparsematrix(
			SGSparseMatrix<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_real_sparsematrix(
			SGSparseMatrix<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	virtual void get_longreal_sparsematrix(
			SGSparseMatrix<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec)=0;
	//@}


	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when loading variable length datatypes
	 * from e.g. file and return the strings and their number
	 * by reference
	 */
	//@{
	virtual void get_bool_string_list(
			SGString<bool>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_byte_string_list(
			SGString<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_int8_string_list(
			SGString<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_char_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_int_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_uint_string_list(
			SGString<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_short_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_word_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_long_string_list(
			SGString<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_ulong_string_list(
			SGString<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_shortreal_string_list(
			SGString<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_real_string_list(
			SGString<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	virtual void get_longreal_string_list(
			SGString<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len)=0;
	//@}

	/** vector access functions */
	/*virtual void get_vector(void*& vector, int32_t& len, DataType& dtype);*/

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when writing vectors of length len
	 * to e.g. a file
	 */
	//@{
	virtual void set_bool_vector(const bool* vector, int32_t len);
	virtual void set_byte_vector(const uint8_t* vector, int32_t len)=0;
	virtual void set_char_vector(const char* vector, int32_t len)=0;
	virtual void set_int_vector(const int32_t* vector, int32_t len)=0;
	virtual void set_shortreal_vector( const float32_t* vector, int32_t len)=0;
	virtual void set_real_vector(const float64_t* vector, int32_t len)=0;
	virtual void set_short_vector(const int16_t* vector, int32_t len)=0;
	virtual void set_word_vector(const uint16_t* vector, int32_t len)=0;
	//@}


	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when writing matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_bool_matrix(
			const bool* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_byte_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_int8_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec)=0;
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
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_bool_sparsematrix(
			const SGSparseMatrix<bool>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_byte_sparsematrix(
			const SGSparseMatrix<uint8_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_int8_sparsematrix(
			const SGSparseMatrix<int8_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_char_sparsematrix(
			const SGSparseMatrix<char>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_int_sparsematrix(
			const SGSparseMatrix<int32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_uint_sparsematrix(
			const SGSparseMatrix<uint32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_long_sparsematrix(
			const SGSparseMatrix<int64_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_ulong_sparsematrix(
			const SGSparseMatrix<uint64_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_short_sparsematrix(
			const SGSparseMatrix<int16_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_word_sparsematrix(
			const SGSparseMatrix<uint16_t>* matrix, int32_t num_feat, int32_t num_vec)=0; 
	virtual void set_shortreal_sparsematrix(
			const SGSparseMatrix<float32_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_real_sparsematrix(
			const SGSparseMatrix<float64_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	virtual void set_longreal_sparsematrix(
			const SGSparseMatrix<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec)=0;
	//@}


	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when writing variable length datatypes
	 * like strings to a file. Here num_str denotes the number of strings
	 * and strings is a pointer to a string structure.
	 */
	//@{
	virtual void set_bool_string_list(
			const SGString<bool>* strings, int32_t num_str);
	virtual void set_byte_string_list(
			const SGString<uint8_t>* strings, int32_t num_str)=0;
	virtual void set_int8_string_list(
			const SGString<int8_t>* strings, int32_t num_str)=0;
	virtual void set_char_string_list(
			const SGString<char>* strings, int32_t num_str)=0;
	virtual void set_int_string_list(
			const SGString<int32_t>* strings, int32_t num_str)=0;
	virtual void set_uint_string_list(
			const SGString<uint32_t>* strings, int32_t num_str)=0;
	virtual void set_short_string_list(
			const SGString<int16_t>* strings, int32_t num_str)=0;
	virtual void set_word_string_list(
			const SGString<uint16_t>* strings, int32_t num_str)=0;
	virtual void set_long_string_list(
			const SGString<int64_t>* strings, int32_t num_str)=0;
	virtual void set_ulong_string_list(
			const SGString<uint64_t>* strings, int32_t num_str)=0;
	virtual void set_shortreal_string_list(
			const SGString<float32_t>* strings, int32_t num_str)=0;
	virtual void set_real_string_list(
			const SGString<float64_t>* strings, int32_t num_str)=0;
	virtual void set_longreal_string_list(
			const SGString<floatmax_t>* strings, int32_t num_str)=0;
	//@}

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
#endif // __FILE_H__
