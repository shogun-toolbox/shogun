/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Evgeniy Andreev (gsomix)
 * Written (W) 2010 Soeren Sonnenburg
 */

#ifndef __LIBSVMFILE_H__
#define __LIBSVMFILE_H__

#include <io/File.h>

#include <io/LineReader.h>
#include <io/Parser.h>
#include <lib/DelimiterTokenizer.h>

namespace shogun
{

/** @brief read sparse real valued features in svm light format
 * e.g. -1 1:10.0 2:100.2 1000:1.3
 * with -1 == (optional) label
 * and dim 1    - value  10.0
 *     dim 2    - value 100.2
 *     dim 1000 - value   1.3
 */
class CLibSVMFile : public CFile
{
public:
	/** default constructor */
	CLibSVMFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CLibSVMFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CLibSVMFile(const char* fname, char rw='r', const char* name=NULL);

	/** destructor */
	virtual ~CLibSVMFile();

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	virtual void get_vector(int8_t*& vector, int32_t& len) { };
	virtual void get_vector(uint8_t*& vector, int32_t& len) { };
	virtual void get_vector(char*& vector, int32_t& len) { };
	virtual void get_vector(int32_t*& vector, int32_t& len) { };
	virtual void get_vector(uint32_t*& vector, int32_t& len) { };
	virtual void get_vector(float64_t*& vector, int32_t& len) { };
	virtual void get_vector(float32_t*& vector, int32_t& len) { };
	virtual void get_vector(floatmax_t*& vector, int32_t& len) { };
	virtual void get_vector(int16_t*& vector, int32_t& len) { };
	virtual void get_vector(uint16_t*& vector, int32_t& len) { };
	virtual void get_vector(int64_t*& vector, int32_t& len) { };
	virtual void get_vector(uint64_t*& vector, int32_t& len) { };
	//@}

	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when loading matrices from e.g. file
	 * and return the matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			int8_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			int64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	virtual void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec) { };
	//@}

	/** @name N-Dimensional Array Access Functions
	 *
	 * Functions to access n-dimensional arrays of one of the several base
	 * data types. These functions are used when loading n-dimensional arrays
	 * from e.g. file and return the them and its dimensions dims and num_dims
	 * by reference
	 */
	//@{
	virtual void get_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims) { };
	virtual void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims) { };
	virtual void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims) { };
	virtual void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims) { };
	virtual void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims){ };
	virtual void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims){ };
	virtual void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims){ };
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_sparse_matrix(
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	virtual void get_sparse_matrix(
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);
	virtual void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec,
			float64_t*& labels, bool load_labels=true);

	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when loading variable length datatypes
	 * from e.g. file and return the strings and their number
	 * by reference
	 */
	//@{
	virtual void get_string_list(
			SGString<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
	virtual void get_string_list(
			SGString<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) { };
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
	virtual void set_vector(const int8_t* vector, int32_t len) { };
	virtual void set_vector(const uint8_t* vector, int32_t len) { };
	virtual void set_vector(const char* vector, int32_t len) { };
	virtual void set_vector(const int32_t* vector, int32_t len) { };
	virtual void set_vector(const uint32_t* vector, int32_t len) { };
	virtual void set_vector(const float32_t* vector, int32_t len) { };
	virtual void set_vector(const float64_t* vector, int32_t len) { };
	virtual void set_vector(const floatmax_t* vector, int32_t len) { };
	virtual void set_vector(const int16_t* vector, int32_t len) { };
	virtual void set_vector(const uint16_t* vector, int32_t len) { };
	virtual void set_vector(const int64_t* vector, int32_t len) { };
	virtual void set_vector(const uint64_t* vector, int32_t len) { };
	//@}

	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when writing matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const uint32_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const int64_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const uint64_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const floatmax_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec) { };
	virtual void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec) { };
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec);
	//@}

	/** @name Sparse Matrix Access Functions With Labels
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	virtual void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec,
			const float64_t* labels);
	//@}


	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when writing variable length datatypes
	 * like strings to a file. Here num_str denotes the number of strings
	 * and strings is a pointer to a string structure.
	 */
	//@{
	virtual void set_string_list(
			const SGString<uint8_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<int8_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<char>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<int32_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<uint32_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<int16_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<uint16_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<int64_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<uint64_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<float32_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<float64_t>* strings, int32_t num_str) { };
	virtual void set_string_list(
			const SGString<floatmax_t>* strings, int32_t num_str) { };
	//@}

	virtual const char* get_name() const { return "LibSVMFile"; }

private:
	/** class initialization */
	void init();

	/** class initialization */
	void init_with_defaults();

	/** get number of lines */
	int32_t get_num_lines();

private:
	/** delimiter for index and data in sparse entries */
	char m_delimiter;

	/** object for reading lines from file */
	CLineReader* m_line_reader;

	/** parser of lines */
	CParser* m_parser;

	/** tokenizer for line_reader */
	CDelimiterTokenizer* m_line_tokenizer;

	/** delimiter for parsing lines */
	CDelimiterTokenizer* m_whitespace_tokenizer;

	/** delimiter for parsing sparse entries */
	CDelimiterTokenizer* m_delimiter_tokenizer;
};

}

#endif /** __LIBSVMFILE_H__ */
