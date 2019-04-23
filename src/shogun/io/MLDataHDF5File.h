/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Liu Zhengyang, Viktor Gal, Thoralf Klein, Abhinav Agarwalla, 
 *          Bjoern Esser
 */
#ifndef __MLDATA_HDF5_FILE_H__
#define __MLDATA_HDF5_FILE_H__

#include <shogun/lib/config.h>

#if defined(HAVE_HDF5) && defined( HAVE_CURL)
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
IGNORE_IN_CLASSLIST class MLDataHDF5File : public File
{
public:
	/** default constructor  */
	MLDataHDF5File();

	/** constructor
	 *
	 * @param fname filename to open
	 * @param rw mode, 'r' or 'w'
	 * @param name variable name (e.g. "x" or "/path/to/x")
	 */
	MLDataHDF5File(char* fname,
                    const char* name=NULL,
                    const char* url_prefix="http://mldata.org/repository/data/download/");

	/** default destructor */
	virtual ~MLDataHDF5File();

	/** @name Vector Access Functions
	 *
	 * Functions to access vectors of one of the several base data types.
	 * These functions are used when loading vectors from e.g. file
	 * and return the vector and its length len by reference
	 */
	//@{
	virtual void get_vector(bool*& vector, int32_t& len);
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
			bool*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_matrix(
			uint8_t*& matrix, int32_t& num_feat, int32_t& num_vec);
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
			SGSparseVector<int8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
	virtual void get_sparse_matrix(
			SGSparseVector<uint8_t>*& matrix, int32_t& num_feat, int32_t& num_vec);
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
			SGVector<bool>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<int8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<uint8_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<char>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<int32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<uint32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<int16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<uint16_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<int64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<uint64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<float32_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<float64_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	virtual void get_string_list(
			SGVector<floatmax_t>*& strings, int32_t& num_str,
			int32_t& max_string_len);
	//@}

	virtual void get_matrix(int8_t*&, int32_t&, int32_t&)
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
	virtual void set_vector(const int8_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const uint8_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const char*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const int16_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const int32_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const uint32_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const float32_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const float64_t*, int32_t){
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const floatmax_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const uint16_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const int64_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_vector(const uint64_t*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const uint8_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const int8_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const char*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const int32_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const uint32_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const int64_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const uint64_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const float32_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const float64_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const floatmax_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const int16_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_matrix(const uint16_t*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<bool>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<uint8_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<int8_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<char>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<int32_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<uint32_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<int64_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<uint64_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<int16_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<uint16_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<float32_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<float64_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_sparse_matrix(const shogun::SGSparseVector<floatmax_t>*, int32_t, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<bool>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<uint8_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<int8_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<char>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<int32_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<uint32_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<int16_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<uint16_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<int64_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<uint64_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<float32_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<float64_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	virtual void set_string_list(const shogun::SGVector<floatmax_t>*, int32_t)
	{
		not_implemented(SOURCE_LOCATION);
	}
	/** @return object name */
	virtual const char* get_name() const { return "HDF5File"; }

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

	/** filename to write the data to */
	char *fname;

	/** constructed url for the data */
	char *mldata_url;
};
}
#endif //  HAVE_CURL && HAVE_HDF5
#endif //__HDF5_FILE_H__

