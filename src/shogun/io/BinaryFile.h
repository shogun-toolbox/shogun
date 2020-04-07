/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Chiyuan Zhang, Viktor Gal, 
 *          Thoralf Klein, Evan Shelhamer
 */
#ifndef __BINARY_FILE_H__
#define __BINARY_FILE_H__

#include <shogun/lib/config.h>
#include <shogun/lib/common.h>
#include <shogun/io/SimpleFile.h>
#include <shogun/io/File.h>

namespace shogun
{
struct TSGDataType;
template <class ST> class SGVector;
template <class T> class SGSparseVector;

/** @brief A Binary file access class.
 *
 * A file consists of a SG00 fourcc header then an alternation of a type header and
 * data. The current implementation is capable of storing only a single
 * header/data type. Multiple headers are currently not implemented.
 */
class BinaryFile: public File
{
public:
	/** default constructor  */
	BinaryFile();

	/** constructor
	 *
	 * @param f already opened file
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	BinaryFile(FILE* f, const char* name=NULL);

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	BinaryFile(const char* fname, char rw='r', const char* name=NULL);

	/** default destructor */
	~BinaryFile() override;

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
#endif // #ifndef SWIG

	/** @return object name */
	const char* get_name() const override { return "BinaryFile"; }

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
		SimpleFile<DT> f(filename, file);
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
		SimpleFile<DT> f(filename, file);
		return f.save(src, num);
	}
};
}
#endif //__BINARY_FILE_H__
