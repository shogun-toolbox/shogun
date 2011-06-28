/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2011 Shashwat Lal Das
 * Copyright (C) 2011 Berlin Institute of Technology and Max-Planck-Society
 */
#ifndef __STREAMING_FILE_H__
#define __STREAMING_FILE_H__

#include "lib/config.h"
#include "base/DynArray.h"
#include "lib/common.h"
#include "lib/File.h"
#include "lib/io.h"

namespace shogun
{
/** @brief A Streaming File access class.
 *
 * - Vectors are read as one vector per line
 * - NOT YET IMPLEMENTED:
 * - Matrices are written out as one column per line
 * - Sparse Matrices are written out as one column per line (libsvm/svmlight
 *	 style format)
 * - Strings are written out as one string per line
 *
 */
	class CStreamingFile: public CFile
	{
	public:
		/** default constructor	 */
		CStreamingFile(void);

		/** constructor
		 *
		 * @param f already opened file
		 * @param name variable name (e.g. "x" or "/path/to/x")
		 */
		CStreamingFile(FILE* f, const char* name=NULL);

		/** constructor
		 *
		 * @param fname filename to open
		 * @param rw mode, 'r' or 'w'
		 * @param name variable name (e.g. "x" or "/path/to/x")
		 */
		CStreamingFile(char* fname, char rw='r', const char* name=NULL);

		/** default destructor */
		virtual ~CStreamingFile();

		/** @name Vector Access Functions
		 *
		 * Functions to access vectors of one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector and its length len by reference
		 */
		//@{
		virtual void get_bool_vector(bool*& vector, int32_t& len);
		virtual void get_byte_vector(uint8_t*& vector, int32_t& len);
		virtual void get_char_vector(char*& vector, int32_t& len);
		virtual void get_int_vector(int32_t*& vector, int32_t& len);
		virtual void get_real_vector(float64_t*& vector, int32_t& len);
		virtual void get_shortreal_vector(float32_t*& vector, int32_t& len);
		virtual void get_short_vector(int16_t*& vector, int32_t& len);
		virtual void get_word_vector(uint16_t*& vector, int32_t& len);
		virtual void get_int8_vector(int8_t*& vector, int32_t& len);
		virtual void get_uint_vector(uint32_t*& vector, int32_t& len);
		virtual void get_long_vector(int64_t*& vector, int32_t& len);
		virtual void get_ulong_vector(uint64_t*& vector, int32_t& len);
		virtual void get_longreal_vector(floatmax_t*& vector, int32_t& len);
		//@}

		/** @name Label and Vector Access Functions
		 *
		 * Functions to access the label and vectors of examples
		 * one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector, its length, and the label by reference
		 */
		//@{
		virtual void get_bool_vector_and_label(bool*& vector, int32_t& len, float64_t& label);
		virtual void get_byte_vector_and_label(uint8_t*& vector, int32_t& len, float64_t& label);
		virtual void get_char_vector_and_label(char*& vector, int32_t& len, float64_t& label);
		virtual void get_int_vector_and_label(int32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_real_vector_and_label(float64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_shortreal_vector_and_label(float32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_short_vector_and_label(int16_t*& vector, int32_t& len, float64_t& label);
		virtual void get_word_vector_and_label(uint16_t*& vector, int32_t& len, float64_t& label);
		virtual void get_int8_vector_and_label(int8_t*& vector, int32_t& len, float64_t& label);
		virtual void get_uint_vector_and_label(uint32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_long_vector_and_label(int64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_ulong_vector_and_label(uint64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_longreal_vector_and_label(floatmax_t*& vector, int32_t& len, float64_t& label);
		//@}

		/** @name Vector Access Functions
		 *
		 * Functions to access vectors of one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector and its length len by reference
		 */
		//@{
		virtual void get_bool_string(bool*& vector, int32_t& len);
		virtual void get_byte_string(uint8_t*& vector, int32_t& len);
		virtual void get_char_string(char*& vector, int32_t& len);
		virtual void get_int_string(int32_t*& vector, int32_t& len);
		virtual void get_real_string(float64_t*& vector, int32_t& len);
		virtual void get_shortreal_string(float32_t*& vector, int32_t& len);
		virtual void get_short_string(int16_t*& vector, int32_t& len);
		virtual void get_word_string(uint16_t*& vector, int32_t& len);
		virtual void get_int8_string(int8_t*& vector, int32_t& len);
		virtual void get_uint_string(uint32_t*& vector, int32_t& len);
		virtual void get_long_string(int64_t*& vector, int32_t& len);
		virtual void get_ulong_string(uint64_t*& vector, int32_t& len);
		virtual void get_longreal_string(floatmax_t*& vector, int32_t& len);
		//@}

		/** @name Label and Vector Access Functions
		 *
		 * Functions to access the label and vectors of examples
		 * one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector, its length, and the label by reference
		 */
		//@{
		virtual void get_bool_string_and_label(bool*& vector, int32_t& len, float64_t& label);
		virtual void get_byte_string_and_label(uint8_t*& vector, int32_t& len, float64_t& label);
		virtual void get_char_string_and_label(char*& vector, int32_t& len, float64_t& label);
		virtual void get_int_string_and_label(int32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_real_string_and_label(float64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_shortreal_string_and_label(float32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_short_string_and_label(int16_t*& vector, int32_t& len, float64_t& label);
		virtual void get_word_string_and_label(uint16_t*& vector, int32_t& len, float64_t& label);
		virtual void get_int8_string_and_label(int8_t*& vector, int32_t& len, float64_t& label);
		virtual void get_uint_string_and_label(uint32_t*& vector, int32_t& len, float64_t& label);
		virtual void get_long_string_and_label(int64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_ulong_string_and_label(uint64_t*& vector, int32_t& len, float64_t& label);
		virtual void get_longreal_string_and_label(floatmax_t*& vector, int32_t& len, float64_t& label);
		//@}

		/** @name Vector Access Functions
		 *
		 * Functions to access vectors of one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector and its length len by reference
		 */
		//@{
		virtual void get_bool_sparse_vector(SGSparseVectorEntry<bool>*& vector, int32_t& len);
		virtual void get_byte_sparse_vector(SGSparseVectorEntry<uint8_t>*& vector, int32_t& len);
		virtual void get_char_sparse_vector(SGSparseVectorEntry<char>*& vector, int32_t& len);
		virtual void get_int_sparse_vector(SGSparseVectorEntry<int32_t>*& vector, int32_t& len);
		virtual void get_real_sparse_vector(SGSparseVectorEntry<float64_t>*& vector, int32_t& len);
		virtual void get_shortreal_sparse_vector(SGSparseVectorEntry<float32_t>*& vector, int32_t& len);
		virtual void get_short_sparse_vector(SGSparseVectorEntry<int16_t>*& vector, int32_t& len);
		virtual void get_word_sparse_vector(SGSparseVectorEntry<uint16_t>*& vector, int32_t& len);
		virtual void get_int8_sparse_vector(SGSparseVectorEntry<int8_t>*& vector, int32_t& len);
		virtual void get_uint_sparse_vector(SGSparseVectorEntry<uint32_t>*& vector, int32_t& len);
		virtual void get_long_sparse_vector(SGSparseVectorEntry<int64_t>*& vector, int32_t& len);
		virtual void get_ulong_sparse_vector(SGSparseVectorEntry<uint64_t>*& vector, int32_t& len);
		virtual void get_longreal_sparse_vector(SGSparseVectorEntry<floatmax_t>*& vector, int32_t& len);
		//@}

		/** @name Vector Access Functions
		 *
		 * Functions to access vectors of one of the several base data types.
		 * These functions are used when loading vectors from e.g. file
		 * and return the vector and its length len by reference
		 */
		//@{
		virtual void get_bool_sparse_vector_and_label(SGSparseVectorEntry<bool>*& vector, int32_t& len, float64_t& label);
		virtual void get_byte_sparse_vector_and_label(SGSparseVectorEntry<uint8_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_char_sparse_vector_and_label(SGSparseVectorEntry<char>*& vector, int32_t& len, float64_t& label);
		virtual void get_int_sparse_vector_and_label(SGSparseVectorEntry<int32_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_real_sparse_vector_and_label(SGSparseVectorEntry<float64_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_shortreal_sparse_vector_and_label(SGSparseVectorEntry<float32_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_short_sparse_vector_and_label(SGSparseVectorEntry<int16_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_word_sparse_vector_and_label(SGSparseVectorEntry<uint16_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_int8_sparse_vector_and_label(SGSparseVectorEntry<int8_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_uint_sparse_vector_and_label(SGSparseVectorEntry<uint32_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_long_sparse_vector_and_label(SGSparseVectorEntry<int64_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_ulong_sparse_vector_and_label(SGSparseVectorEntry<uint64_t>*& vector, int32_t& len, float64_t& label);
		virtual void get_longreal_sparse_vector_and_label(SGSparseVectorEntry<floatmax_t>*& vector, int32_t& len, float64_t& label);
		//@}

		/** @return object name */
		inline virtual const char* get_name() const { return "StreamingFile"; }
	};
}
#endif //__STREAMING_FILE_H__
