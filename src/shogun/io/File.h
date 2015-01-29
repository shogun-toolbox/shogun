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

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>

namespace shogun
{
template <class ST> class SGString;
template <class ST> class SGSparseVector;
template <class ST> struct SGSparseVectorEntry;

/** @brief A File access base class.
 *
 * A file is assumed to be a seekable raw data stream.
 *
 * \sa CCSVFile
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

#ifdef HAVE_FDOPEN 
	/** constructor
	 *
	 * @param fd already opened file descriptor
	 * @param mode mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CFile(int fd, const char* mode, const char* name=NULL);
#endif

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	CFile(const char* fname, char rw='r', const char* name=NULL);

	/** default destructor */
	virtual ~CFile();

	/** close */
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

#ifndef SWIG // SWIG should skip this
	/** get the file descriptor
	 *
	 * @return FILE ptr
	 */
	FILE* get_file_descriptor()
	{
		return file;
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
	virtual void get_vector(bool*& vector, int32_t& len);
	virtual void get_vector(int8_t*& vector, int32_t& len){};
	virtual void get_vector(uint8_t*& vector, int32_t& len){};
	virtual void get_vector(char*& vector, int32_t& len){};
	virtual void get_vector(int32_t*& vector, int32_t& len){};
	virtual void get_vector(uint32_t*& vector, int32_t& len){};
	virtual void get_vector(float64_t*& vector, int32_t& len){};
	virtual void get_vector(float32_t*& vector, int32_t& len){};
	virtual void get_vector(floatmax_t*& vector, int32_t& len){};
	virtual void get_vector(int16_t*& vector, int32_t& len){};
	virtual void get_vector(uint16_t*& vector, int32_t& len){};
	virtual void get_vector(int64_t*& vector, int32_t& len){};
	virtual void get_vector(uint64_t*& vector, int32_t& len){};
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
	virtual void get_matrix(
			bool*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			int8_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			int64_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec){};
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
	virtual void get_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims){};
	virtual void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims){};
	//@}
	//
	/** @name Sparse Vector Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse vectors from e.g. file
	 * and return the sparse vectors and its length num_feat by reference
	 */
	//@{
	virtual void get_sparse_vector(
			SGSparseVectorEntry<bool>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<uint8_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<int8_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<char>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<int32_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<uint32_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<int64_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<uint64_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<int16_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<uint16_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<float32_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<float64_t>*& entries, int32_t& num_feat);
	virtual void get_sparse_vector(
			SGSparseVectorEntry<floatmax_t>*& entries, int32_t& num_feat);
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
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
		SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
	virtual void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec){};
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
			SGString<bool>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGString<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<char>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
	virtual void get_string_list(
			SGString<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len){};
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
	virtual void set_vector(const bool* vector, int32_t len);
	virtual void set_vector(const int8_t* vector, int32_t len){};
	virtual void set_vector(const uint8_t* vector, int32_t len){};
	virtual void set_vector(const char* vector, int32_t len){};
	virtual void set_vector(const int32_t* vector, int32_t len){};
	virtual void set_vector(const uint32_t* vector, int32_t len){};
	virtual void set_vector(const float32_t* vector, int32_t len){};
	virtual void set_vector(const float64_t* vector, int32_t len){};
	virtual void set_vector(const floatmax_t* vector, int32_t len){};
	virtual void set_vector(const int16_t* vector, int32_t len){};
	virtual void set_vector(const uint16_t* vector, int32_t len){};
	virtual void set_vector(const int64_t* vector, int32_t len){};
	virtual void set_vector(const uint64_t* vector, int32_t len){};
	//@}


	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when writing matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_matrix(
			const bool* matrix, int32_t num_feat, int32_t num_vec);
	virtual void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const uint32_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const int64_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const uint64_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const floatmax_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec){};
	//@}
	//
	/** @name Sparse Vector Access Functions
	 *
	 * Functions to access sparse vectors of one of the several base data types.
	 * These functions are used when writing sparse vectors of num_feat entries
	 * to e.g. a file
	 */
	//@{
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<bool>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<uint8_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<int8_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<char>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<int32_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<uint32_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<int64_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<uint64_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<int16_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<uint16_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<float32_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<float64_t>* entries, int32_t num_feat);
	virtual void set_sparse_vector(
			const SGSparseVectorEntry<floatmax_t>* entries, int32_t num_feat);
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	virtual void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec){};
	virtual void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec){};
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
			const SGString<bool>* strings, int32_t num_str);
	virtual void set_string_list(
			const SGString<uint8_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<int8_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<char>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<int32_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<uint32_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<int16_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<uint16_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<int64_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<uint64_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<float32_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<float64_t>* strings, int32_t num_str){};
	virtual void set_string_list(
			const SGString<floatmax_t>* strings, int32_t num_str){};
	//@}

	/** @return object name */
	virtual const char* get_name() const { return "File"; }

    /** read whole file in memory
     *
     * @param fname - file name
     * @param len - length of file (returned by reference)
     * @return buffer to read file - needs to be freed with SG_FREE
     */
    static char* read_whole_file(char* fname, size_t& len);
#endif // #ifndef SWIG

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
