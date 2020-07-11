/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Evgeniy Andreev, Sergey Lisitsyn, 
 *          Heiko Strathmann, Yuyu Zhang, Viktor Gal, Bjoern Esser, 
 *          Saurabh Goyal
 */

#ifndef __PROTOBUFFILE_H__
#define __PROTOBUFFILE_H__

#include <shogun/lib/config.h>

#ifdef HAVE_PROTOBUF

#include <shogun/io/File.h>

// a hack to avoid clashes with apple's ConditionalMacros.h
#ifdef __APPLE__
    #ifdef TYPE_BOOL
        #define ___APPLE_TYPE_BOOL TYPE_BOOL
        #undef TYPE_BOOL
    #endif
#endif

#ifdef __APPLE__
    #ifdef ___APPLE_TYPE_BOOL
        #define TYPE_BOOL ___APPLE_TYPE_BOOL
        #undef ___APPLE_TYPE_BOOL
    #endif
#endif

#include <shogun/io/protobuf/ShogunVersion.pb.h>
#include <shogun/io/protobuf/Headers.pb.h>
#include <shogun/io/protobuf/Chunks.pb.h>

namespace shogun
{

/** @brief Class for work with binary file
 * in protobuf format.
 *
 * Format of serialized data in byte file:
 * <pre>
 * size of ShogunVersion message - big endian uint32
 * ShogunVersion message
 * size of next message - big endian uint32
 * data message, e.g. Int32Chunk
 * ...
 * </pre>
 */
class ProtobufFile : public File
{
public:
	/** default constructor */
	ProtobufFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	ProtobufFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	ProtobufFile(const char* fname, char rw='r', const char* name=NULL);

	/** destructor */
	~ProtobufFile() override;

#ifndef SWIG // SWIG should skip this
	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	void get_vector(int8_t*& vector, int32_t& len) override;
	void get_vector(uint8_t*& vector, int32_t& len) override;
	void get_vector(char*& vector, int32_t& len) override;
	void get_vector(int32_t*& vector, int32_t& len) override;
	void get_vector(uint32_t*& vector, int32_t& len) override;
	void get_vector(float64_t*& vector, int32_t& len) override;
	void get_vector(float32_t*& vector, int32_t& len) override;
	void get_vector(floatmax_t*& vector, int32_t& len) override;
	void get_vector(int16_t*& vector, int32_t& len) override;
	void get_vector(uint16_t*& vector, int32_t& len) override;
	void get_vector(int64_t*& vector, int32_t& len) override;
	void get_vector(uint64_t*& vector, int32_t& len) override;
	//@}

	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when loading matrices from e.g. file
	 * and return the matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			int8_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			char*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			int32_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			uint32_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			int64_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			uint64_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			float32_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			float64_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			floatmax_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			int16_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			uint16_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	//@}

	/** @name N-Dimensional Array Access Functions
	 *
	 * Functions to access n-dimensional arrays of one of the several base
	 * data types. These functions are used when loading n-dimensional arrays
	 * from e.g. file and return the them and its dimensions dims and num_dims
	 * by reference
	 */
	//@{
	void get_ndarray(
			uint8_t*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			char*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			int32_t*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			float32_t*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			float64_t*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			int16_t*& array, int32_t*& dims, int32_t& num_dims) override;
	void get_ndarray(
			uint16_t*& array, int32_t*& dims, int32_t& num_dims) override;
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when loading sparse matrices from e.g. file
	 * and return the sparse matrices and its dimensions num_feat and num_vec
	 * by reference
	 */
	//@{
	void get_sparse_matrix(
			SGSparseVector<bool>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
		SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<char>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<int32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<uint32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<int64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<uint64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<int16_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<uint16_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<float32_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<float64_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<floatmax_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	//@}

	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when loading variable length datatypes
	 * from e.g. file and return the strings and their number
	 * by reference
	 */
	//@{
	void get_string_list(
			SGVector<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<char>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
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
	void set_vector(const int8_t* vector, int32_t len) override;
	void set_vector(const uint8_t* vector, int32_t len) override;
	void set_vector(const char* vector, int32_t len) override;
	void set_vector(const int32_t* vector, int32_t len) override;
	void set_vector(const uint32_t* vector, int32_t len) override;
	void set_vector(const float32_t* vector, int32_t len) override;
	void set_vector(const float64_t* vector, int32_t len) override;
	void set_vector(const floatmax_t* vector, int32_t len) override;
	void set_vector(const int16_t* vector, int32_t len) override;
	void set_vector(const uint16_t* vector, int32_t len) override;
	void set_vector(const int64_t* vector, int32_t len) override;
	void set_vector(const uint64_t* vector, int32_t len) override;
	//@}

	/** @name Matrix Access Functions
	 *
	 * Functions to access matrices of one of the several base data types.
	 * These functions are used when writing matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const char* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const int32_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const uint32_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const int64_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const uint64_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const float32_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const float64_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const floatmax_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const int16_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const uint16_t* matrix, int32_t num_feat, int32_t num_vec) override;
	//@}

	/** @name Sparse Matrix Access Functions
	 *
	 * Functions to access sparse matrices of one of the several base data types.
	 * These functions are used when writing sparse matrices of num_feat rows and
	 * num_vec columns to e.g. a file
	 */
	//@{
	void set_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	//@}

	/** @name String Access Functions
	 *
	 * Functions to access strings of one of the several base data types.
	 * These functions are used when writing variable length datatypes
	 * like strings to a file. Here num_str denotes the number of strings
	 * and strings is a pointer to a string structure.
	 */
	//@{
	void set_string_list(
			const SGVector<uint8_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<int8_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<char>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<int32_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<uint32_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<int16_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<uint16_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<int64_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<uint64_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<float32_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<float64_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<floatmax_t>* strings, int32_t num_str) override;
	//@}
#endif // #ifndef SWIG // SWIG should skip this

	const char* get_name() const override { return "ProtobufFile"; }

private:
	/** class initialization */
	void init();

	/** write unsigned int to array in big endian format */
	void write_big_endian_uint(uint32_t number, uint8_t* array, uint32_t size);

	/** read unsigned int from array in big endian format */
	uint32_t read_big_endian_uint(uint8_t* array, uint32_t size);

	/** compute number of messages for storing data */
	int32_t compute_num_messages(uint64_t len, int32_t sizeof_type) const;

	/** read global header */
	void read_and_validate_global_header(ShogunVersion_SGDataType type);

	/** write global header */
	void write_global_header(ShogunVersion_SGDataType type);

	/** read data type headers */
	//@{
	VectorHeader read_vector_header();
	MatrixHeader read_matrix_header();
	SparseMatrixHeader read_sparse_matrix_header();
	StringListHeader read_string_list_header();
	//@}

	/** write data type headers */
	//@{
	void write_vector_header(int32_t len, int32_t num_messages);
	void write_matrix_header(int32_t num_feat, int32_t num_vec, int32_t num_messages);
	//@}

	/** write sparse matrix header with information
	 * about length of each sparse vector (row)
	 */
	//@{
	void write_sparse_matrix_header(
			const SGSparseVector<bool>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<char>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<int32_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<uint32_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<float64_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<float32_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<int16_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<uint16_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<int64_t>* matrix, int32_t num_feat, int32_t num_vec);
	void write_sparse_matrix_header(
			const SGSparseVector<uint64_t>* matrix, int32_t num_feat, int32_t num_vec);
	//@}

	/** write string list header with information
	 * about length of each string
	 */
	//@{
	void write_string_list_header(
			const SGVector<uint8_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<int8_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<char>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<int32_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<uint32_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<float64_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<float32_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<floatmax_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<int16_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<uint16_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<int64_t>* strings, int32_t num_str);
	void write_string_list_header(
			const SGVector<uint64_t>* strings, int32_t num_str);
	//@}

	/** read message */
	void read_message(google::protobuf::Message& message);

	/** write message */
	void write_message(const google::protobuf::Message& message);

	/** @name Memory Block Access Functions
	 *
	 * Read chunks from protobuf binary file
	 */
	//@{
	void read_memory_block(uint8_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(int8_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(char*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(int32_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(uint32_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(float64_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(float32_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(floatmax_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(int16_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(uint16_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(int64_t*& vector, uint64_t len, int32_t num_messages);
	void read_memory_block(uint64_t*& vector, uint64_t len, int32_t num_messages);
	//@}

	/** @name Memory Block Access Functions
	 *
	 * Write chunks to protobuf binary file
	 */
	//@{
	void write_memory_block(const int8_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const uint8_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const char* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const int32_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const uint32_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const float32_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const float64_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const floatmax_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const int16_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const uint16_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const int64_t* vector, uint64_t len, int32_t num_messages);
	void write_memory_block(const uint64_t* vector, uint64_t len, int32_t num_messages);
	//@}

	//@{
	void read_sparse_matrix(SGSparseVector<bool>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<uint8_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<int8_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<char>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<int32_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<uint32_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<int16_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<uint16_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<int64_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<uint64_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<float32_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<float64_t>*& matrix,
			const SparseMatrixHeader& data_header);
	void read_sparse_matrix(SGSparseVector<floatmax_t>*& matrix,
			const SparseMatrixHeader& data_header);
	//@}

	/**
	 */
	//@{
	void write_sparse_matrix(
			const SGSparseVector<bool>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<int8_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<char>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<int32_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<uint32_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<int16_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<uint16_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<int64_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<uint64_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<float32_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<float64_t>* matrix, int32_t num_vec);
	void write_sparse_matrix(
			const SGSparseVector<floatmax_t>* matrix, int32_t num_vec);
	//@}

	//@{
	void read_string_list(SGVector<uint8_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<int8_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<char>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<int32_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<uint32_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<int16_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<uint16_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<int64_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<uint64_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<float32_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<float64_t>*& strings,
			const StringListHeader& data_header);
	void read_string_list(SGVector<floatmax_t>*& strings,
			const StringListHeader& data_header);
	//@}

	/**
	 */
	//@{
	void write_string_list(
			const SGVector<uint8_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<int8_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<char>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<int32_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<uint32_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<int16_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<uint16_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<int64_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<uint64_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<float32_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<float64_t>* strings, int32_t num_str);
	void write_string_list(
			const SGVector<floatmax_t>* strings, int32_t num_str);
	//@}

private:
	/** version of protobuf file */
	int32_t version;

	/** maximal size of single message */
	int32_t message_size;

	/** byte buffer */
	uint8_t* buffer;

	/** buffer for numbers */
	uint8_t uint_buffer[4];
};

}

#endif /** HAVE_PROTOBUF */
#endif /** __PROTOBUFFILE_H__ */
