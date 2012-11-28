/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2010 Soeren Sonnenburg
 * Copyright (C) 2010 Berlin Institute of Technology
 */
#ifndef __BINARY_FILE_H__
#define __BINARY_FILE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/base/SGObject.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/SimpleFile.h>
#include <shogun/io/File.h>

namespace shogun
{
/** @brief A Binary file access class.
 *
 * A file consists of a SG00 fourcc header then an alternation of a type header and
 * data. The current implementation is capable of storing only a single
 * header/data type. Multiple headers are currently not implemented.
 */
class CBinaryFile: public CFile
{
public:
	/** default constructor  */
	CBinaryFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CBinaryFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CBinaryFile(const char* fname, char rw='r', const char* name=NULL);

	/** default destructor */
	virtual ~CBinaryFile();

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	virtual void get_vector(int8_t*& vector, int32_t& len);
	virtual void get_vector(uint8_t*& vector, int32_t& len);
	virtual void get_vector(char*& vector, int32_t& len);
	virtual void get_vector(int32_t*& vector, int32_t& len);
	virtual void get_vector(uint32_t*& vector, int32_t& len);
	virtual void get_vector(float64_t*& vector, int32_t& len);
	virtual void get_vector(float32_t*& vector, int32_t& len);
	virtual void get_vector(floatmax_t*& vector, int32_t& len);
	virtual void get_vector(int16_t*& vector, int32_t& len);
	virtual void get_vector(uint16_t*& vector, int32_t& len);
	virtual void get_vector(int64_t*& vector, int32_t& len);
	virtual void get_vector(uint64_t*& vector, int32_t& len);
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
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			int8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			int64_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec);
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
			uint8_t*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims);
	virtual void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims);
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
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	//@}

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when writing vectors of length len
	 * to e.g. a file
	 */
	//@{
	virtual void set_vector(const int8_t* vector, int32_t len);
	virtual void set_vector(const uint8_t* vector, int32_t len);
	virtual void set_vector(const char* vector, int32_t len);
	virtual void set_vector(const int32_t* vector, int32_t len);
	virtual void set_vector(const uint32_t* vector, int32_t len);
	virtual void set_vector(const float32_t* vector, int32_t len);
	virtual void set_vector(const float64_t* vector, int32_t len);
	virtual void set_vector(const floatmax_t* vector, int32_t len);
	virtual void set_vector(const int16_t* vector, int32_t len);
	virtual void set_vector(const uint16_t* vector, int32_t len);
	virtual void set_vector(const int64_t* vector, int32_t len);
	virtual void set_vector(const uint64_t* vector, int32_t len);
	//@}


	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when writing matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const uint32_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const int64_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const uint64_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const floatmax_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec);
	//@}

	/** @name N-Dimensional Array Access Functions
	 *
	 * Functions to access n-dimensional arrays of one of the several base data types.
	 * These functions are used when writing array of num_dims dimensions to e.g. a file.
	 * Dims contain sizes of every dimensions.
	 */
	//@{
	virtual void set_ndarray(
			const uint8_t* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const char* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const int32_t* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const float32_t* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const float64_t* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const int16_t* array, int32_t* dims, int32_t num_dims);
	virtual void set_ndarray(
			const uint16_t* array, int32_t* dims, int32_t num_dims);
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


	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when writing variable length datatypes
	 * like strings to a file. Here num_str denotes the number of strings
	 * and strings is a pointer to a string structure.
	 */
	//@{
	virtual void set_string_list(
			const SGString<uint8_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<int8_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<char>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<int32_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<uint32_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<int16_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<uint16_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<int64_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<uint64_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<float32_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<float64_t>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<floatmax_t>* strings, int32_t num_str);
	//@}

	/** @return object name */
	virtual const char* get_name() const { return "BinaryFile"; }

protected:
    /** read header
	 *
     * @return datatype
     */
    void read_header(TSGDataType* dest);

    /** write header
	 *
     * @param datatype we are writing
     */
    void write_header(const TSGDataType* datatype);

    /** parse first header - defunct!
     *
     * @param type feature type
     * @return -1
     */
    int32_t parse_first_header(TSGDataType& type);

    /** parse next header - defunct!
     *
     * @param type feature type
     * @return -1
     */
    int32_t parse_next_header(TSGDataType& type);

private:
	/** load data (templated)
	 *
	 * @param target loaded data
	 * @param num number of data elements
	 * @return loaded data
	 */
	template <class DT> DT* load_data(DT* target, int64_t& num)
	{
		CSimpleFile<DT> f(filename, file);
		return f.load(target, num);
	}

	/** save data (templated)
	 *
	 * @param src data to save
	 * @param num number of data elements
	 * @return whether operation was successful
	 */
	template <class DT> bool save_data(DT* src, int64_t num)
	{
		CSimpleFile<DT> f(filename, file);
		return f.save(src, num);
	}
};
}
#endif //__BINARY_FILE_H__
