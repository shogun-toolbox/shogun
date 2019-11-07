/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Soumyajit De, Sergey Lisitsyn, Yingrui Chang,
 *          Evgeniy Andreev, Yuyu Zhang, Viktor Gal, Thoralf Klein,
 *          Fernando Iglesias, Bjoern Esser
 */

#ifndef __SGSPARSEMATRIX_H__
#define __SGSPARSEMATRIX_H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/SGReferencedData.h>
#include <shogun/lib/SGVector.h>
#include <shogun/io/SGIO.h>

namespace shogun
{

template <class T> class SGSparseVector;
template <class ST> struct SGSparseVectorEntry;
template<class T> class SGMatrix;
class File;
class LibSVMFile;

/** @brief template class SGSparseMatrix */
template <class T> class SGSparseMatrix : public SGReferencedData
{
	public:
		/** default constructor */
		SGSparseMatrix();

		/** constructor for setting params */
		SGSparseMatrix(SGSparseVector<T>* vecs, index_t num_feat,
				index_t num_vec, bool ref_counting=true);

		/** constructor to create new matrix in memory */
		SGSparseMatrix(index_t num_feat, index_t num_vec, bool ref_counting=true);

		/** constructor to create new sparse matrix from a dense one
		 *
		 * @param dense dense matrix to be converted
		 */
		SGSparseMatrix(SGMatrix<T> dense);

		/** copy constructor */
		SGSparseMatrix(const SGSparseMatrix &orig);

		/** destructor */
		virtual ~SGSparseMatrix();

		/** index access operator */
		inline const SGSparseVector<T>& operator[](index_t index) const
		{
			return sparse_matrix[index];
		}

		/** index access operator */
		inline SGSparseVector<T>& operator[](index_t index)
		{
			return sparse_matrix[index];
		}

		/**
		 * get the sparse matrix (no copying is done here)
		 *
		 * @return the refcount increased matrix
		 */
		inline SGSparseMatrix<T> get()
		{
			return *this;
		}

		/** compute sparse-matrix dense-vector multiplication
		 * @param v the dense-vector to be multiplied with
		 * @return the result vector \f$Q*v\f$, Q being this sparse matrix
		 */
		const SGVector<T> operator*(SGVector<T> v) const
		{
			SGVector<T> result(num_vectors);
			require(v.vlen==num_features,
				"Dimension mismatch! {} vs {}",
				v.vlen, num_features);
			for (index_t i=0; i<num_vectors; ++i)
				result[i]=sparse_matrix[i].dense_dot(1.0, v.vector, v.vlen, 0.0);

			return result;
		}

		/** compute sparse-matrix dense-vector multiplication
		 * @param v the dense-vector to be multiplied with
		 * @return the result vector \f$Q*v\f$, Q being this sparse matrix
		 */
		template<class ST> const SGVector<T> operator*(SGVector<ST> v) const;

		/** operator overload for sparse-matrix read only access
		 * @param i_row
		 * @param i_col
		 */
		inline const T operator()(index_t i_row, index_t i_col) const
		{
			require(i_row>=0, "Provided row index {} negative!", i_row);
			require(i_col>=0, "Provided column index {} negative!", i_col);
			require(i_row<num_features, "Provided row index ({}) is larger than number of rows ({})",
							i_row, num_features);
			require(i_col<num_vectors, "Provided column index ({}) is larger than number of columns ({})",
							i_col, num_vectors);

			for (index_t i=0; i<sparse_matrix[i_col].num_feat_entries; ++i)
			{
				if (i_row==sparse_matrix[i_col].features[i].feat_index)
					return sparse_matrix[i_col].features[i].entry;
			}
			return 0;
		}

		/** operator overload for sparse-matrix r/w access
		 * @param i_row
		 * @param i_col
		 */
		inline T& operator()(index_t i_row, index_t i_col)
		{
			require(i_row>=0, "Provided row index {} negative!", i_row);
			require(i_col>=0, "Provided column index {} negative!", i_col);
			require(i_row<num_features, "Provided row index ({}) is larger than number of rows ({})",
							i_row, num_features);
			require(i_col<num_vectors, "Provided column index ({}) is larger than number of columns ({})",
							i_col, num_vectors);

			for (index_t i=0; i<sparse_matrix[i_col].num_feat_entries; ++i)
			{
				if (i_row==sparse_matrix[i_col].features[i].feat_index)
					return sparse_matrix[i_col].features[i].entry;
			}
			index_t j=sparse_matrix[i_col].num_feat_entries;
			sparse_matrix[i_col].num_feat_entries=j+1;
			sparse_matrix[i_col].features=SG_REALLOC(SGSparseVectorEntry<T>,
				sparse_matrix[i_col].features, j, j+1);
			sparse_matrix[i_col].features[j].feat_index=i_row;
			sparse_matrix[i_col].features[j].entry=static_cast<T>(0);
			return sparse_matrix[i_col].features[j].entry;
		}

		/** load sparse matrix from file
		 *
		 * @param loader File object via which to load data
		 */
		void load(const std::shared_ptr<File>& loader);

		/** load sparse matrix from libsvm file together with labels
		 *
		 * @param libsvm_file the libsvm file
		 * @param do_sort_features whether to sort the vector indices (such that they are in
		 * ascending order) after loading
		 * @return label vector
		 */
		SGVector<float64_t> load_with_labels(const std::shared_ptr<LibSVMFile>& libsvm_file, bool do_sort_features=true);

		/** save sparse matrix to file
		 *
		 * @param saver File object via which to save data
		 */
		void save(const std::shared_ptr<File>& saver);

		/** save sparse matrix together with labels to file
		 *
		 * @param saver File object via which to save data
		 * @param labels label vector
		 */
		void save_with_labels(const std::shared_ptr<LibSVMFile>& saver, SGVector<float64_t> labels);

		/** return the transposed of the sparse matrix */
		SGSparseMatrix<T> get_transposed();

		/** create a sparse matrix from a dense one
		 *
		 * @param full the dense matrix to create the sparse one from
		 */
		void from_dense(SGMatrix<T> full);

		/** sort the indices of the sparse matrix such that they are in ascending order */
		void sort_features();

		/** Pointer identify comparison.
		 *  @return true iff number of vectors and features and pointer are
		 * equal
		 */
		bool operator==(const SGSparseMatrix<T>& other) const;

		/** Equals method up to precision for matrix (element-wise)
		 * @param other matrix to compare with
		 * @return false if any element differs or if shapes are different,
		 * true otherwise
		 */
		bool equals(const SGSparseMatrix<T>& other) const;

	protected:
		/** copy data */
		virtual void copy_data(const SGReferencedData& orig);

		/** init data */
		virtual void init_data();

		/** free data */
		virtual void free_data();

public:

	/// total number of vectors
	index_t num_vectors;

	/// total number of features
	index_t num_features;

	/// array of sparse vectors of size num_vectors
	SGSparseVector<T>* sparse_matrix;

};
}
#endif // __SGSPARSEMATRIX_H__
