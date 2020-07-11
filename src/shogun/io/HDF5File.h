/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Viktor Gal, Thoralf Klein, 
 *          Evan Shelhamer, Abhinav Agarwalla
 */
#ifndef __HDF5_FILE_H__
#define __HDF5_FILE_H__

#include <shogun/lib/config.h>

#ifdef HAVE_HDF5
#include <shogun/lib/common.h>
#include <shogun/io/SGIO.h>
#include <shogun/io/File.h>
#include <hdf5.h>


namespace shogun
{
template <class ST> class SGVector;
template <class ST> class SGSparseVector;
struct TSGDataType;

/** @brief A HDF5 File access class.
 *
 * This class allows reading and writing of vectors and matrices
 * in the hierarchical file format version 5.
 *
 */
#define IGNORE_IN_CLASSLIST
IGNORE_IN_CLASSLIST class HDF5File : public File
{
public:
	/** default constructor  */
	HDF5File();

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	HDF5File(char* fname, char rw='r', const char* name=NULL);

	/** default destructor */
	~HDF5File() override;

#ifndef SWIG // SWIG should skip this
	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	void get_vector(bool*& vector, int32_t& len) override;
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
			bool*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec) override;
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
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
	void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec) override;
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
			SGVector<bool>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len) override;
	void get_string_list(
			SGVector<uint8_t>*& strings, int32_t& num_str,
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
	void set_vector(const bool* vector, int32_t len) override;
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
			const bool* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const int8_t* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_matrix(
			const uint8_t* matrix, int32_t num_feat, int32_t num_vec) override;
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
			const SGSparseVector<int8_t>* matrix, int32_t num_feat, int32_t num_vec) override;
	void set_sparse_matrix(
			const SGSparseVector<uint8_t>* matrix, int32_t num_feat, int32_t num_vec) override;
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
			const SGVector<bool>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<int8_t>* strings, int32_t num_str) override;
	void set_string_list(
			const SGVector<uint8_t>* strings, int32_t num_str) override;
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

	void get_matrix(int8_t*&, int32_t&, int32_t&) override
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void get_int8_sparsematrix(shogun::SGSparseVector<signed char>*&, int32_t&, int32_t&)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void get_int8_string_list(shogun::SGVector<signed char>*&, int32_t&, int32_t&)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_int8_matrix(const int8_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_int8_sparsematrix(const shogun::SGSparseVector<signed char>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_int8_string_list(const shogun::SGVector<signed char>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
#endif // #ifndef SWIG // SWIG should skip this

	/** @return object name */
	const char* get_name() const override { return "HDF5File"; }

protected:
	/** determine the hdf5 type compatible to 'bool' */
	void get_boolean_type();

	/** determine the hdf5 type of class t_class that is
	 * compatible to datatype
	 *
	 * @param t_class hdf5 class
	 * @param datatype shogun file data type
	 *
	 * @return compatible hdf5 datatype or -1
	 */
	hid_t get_compatible_type(H5T_class_t t_class,
							  const TSGDataType* datatype);

	/** get dimensionality of the data
	 *
	 * @param dataset hdf5 dataset
	 * @param dims dimensions (returned by reference)
	 * @param ndims (returned by reference)
	 * @param total_elements (returned by reference)
	 */
	void get_dims(hid_t dataset, int32_t*& dims, int32_t& ndims, int64_t& total_elements);

	/** create a group hierarchy in the hdf5 file h5file according to name */
	void create_group_hierarchy();

protected:
	/** hdf5 file handle */
	hid_t h5file;
	/** hdf5 type closest to 'bool' */
	hid_t boolean_type;
};
}
#endif //  HAVE_HDF5
#endif //__HDF5_FILE_H__

