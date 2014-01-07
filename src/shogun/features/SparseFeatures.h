/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2012 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _SPARSEFEATURES__H__
#define _SPARSEFEATURES__H__

#include <lib/common.h>
#include <lib/DataType.h>
#include <lib/SGSparseMatrix.h>
#include <lib/Cache.h>
#include <io/File.h>
#include <io/LibSVMFile.h>

#include <labels/RegressionLabels.h>
#include <features/Features.h>
#include <features/DotFeatures.h>
#include <features/DenseFeatures.h>

namespace shogun
{

class CFile;
class CLibSVMFile;
class CRegressionLabels;
class CFeatures;
class CDotFeatures;
template <class ST> class CDenseFeatures;
template <class ST> class SGSparseMatrix;

/** @brief Template class SparseFeatures implements sparse matrices.
 *
 * Features are an array of SGSparseVector. Within each vector feat_index are
 * sorted (increasing).
 *
 * Sparse feature vectors can be accessed via get_sparse_feature_vector() and
 * should be freed (this operation is a NOP in most cases) via
 * free_sparse_feature_vector().
 *
 * As this is a template class it can directly be used for different data types
 * like sparse matrices of real valued, integer, byte etc type.
 *
 * (Partly) subset access is supported for this feature type.
 * Simple use the (inherited) add_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method.
 * See also CFeatures class documentation
 */
template <class ST> class CSparseFeatures : public CDotFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 */
		CSparseFeatures(int32_t size=0);

		/** convenience constructor that creates sparse features from
		 * sparse features
		 *
		 * @param sparse sparse matrix
		 */
		CSparseFeatures(SGSparseMatrix<ST> sparse);

		/** convenience constructor that creates sparse features from
		 * dense features
		 *
		 * @param dense dense feature matrix
		 */
		CSparseFeatures(SGMatrix<ST> dense);

		/** copy constructor */
		CSparseFeatures(const CSparseFeatures & orig);

		/** constructor loading features from file
		 *
		 * @param loader File object to load data from
		 */
		CSparseFeatures(CFile* loader);

		/** default destructor */
		virtual ~CSparseFeatures();

		/** free sparse feature matrix
		 *
		 * any subset is removed
		 */
		void free_sparse_feature_matrix();

		/** free sparse feature matrix and cache
		 *
		 * any subset is removed
		 */
		void free_sparse_features();

		/** duplicate feature object
		 *
		 * @return feature object
		 */
		virtual CFeatures* duplicate() const;

		/** get a single feature
		 *
		 * possible with subset
		 *
		 * @param num number of feature vector to retrieve
		 * @param index index of feature in this vector
		 *
		 * @return sum of features that match dimension index and 0 if none is found
		 */
		ST get_feature(int32_t num, int32_t index);

		/** get the fully expanded dense feature vector num
		  *
		  * @return dense feature vector
		  * @param num index of feature vector
		  */
		SGVector<ST> get_full_feature_vector(int32_t num);

		/** get number of non-zero features in vector
		 *
		 * @param num which vector
		 * @return number of non-zero features in vector
		 */
		virtual int32_t get_nnz_features_for_vector(int32_t num);

		/** get sparse feature vector
		 * for sample num from the matrix as it is if matrix is initialized,
		 * else return preprocessed compute_feature_vector
		 *
		 * possible with subset
		 *
		 * @param num index of feature vector
		 * @return sparse feature vector
		 */
		SGSparseVector<ST> get_sparse_feature_vector(int32_t num);

		/** compute the dot product between dense weights and a sparse feature vector
		 * alpha * sparse^T * w + b
		 *
		 * possible with subset
		 *
		 * @param alpha scalar to multiply with
		 * @param num index of feature vector
		 * @param vec dense vector to compute dot product with
		 * @param dim length of the dense vector
		 * @param b bias
		 * @return dot product between dense weights and a sparse feature vector
		 */
		ST dense_dot(ST alpha, int32_t num, ST* vec, int32_t dim, ST b);

		/** add a sparse feature vector onto a dense one
		 * dense+=alpha*sparse
		 *
		 * possible with subset
		 *
		 @param alpha scalar to multiply with
		 @param num index of feature vector
		 @param vec dense vector
		 @param dim length of the dense vector
		 @param abs_val if true, do dense+=alpha*abs(sparse)
		 */
		void add_to_dense_vec(float64_t alpha, int32_t num,
				float64_t* vec, int32_t dim, bool abs_val=false);

		/** free sparse feature vector
		 *
		 * possible with subset
		 *
		 * @param num index of this vector in the cache
		 */
		void free_sparse_feature_vector(int32_t num);

		/** get the pointer to the sparse feature matrix
		 * num_feat,num_vectors are returned by reference
		 *
		 * not possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return feature matrix
		 */
		SGSparseVector<ST>* get_sparse_feature_matrix(int32_t &num_feat, int32_t &num_vec);

		/** get the sparse feature matrix
		 *
		 * not possible with subset
		 *
		 * @return sparse matrix
		 *
		 */
		SGSparseMatrix<ST> get_sparse_feature_matrix();

		/** get a transposed copy of the features
		 *
		 * not possible with subset
		 *
		 * @return transposed copy
		 */
		CSparseFeatures<ST>* get_transposed();

		/** compute and return the transpose of the sparse feature matrix
		 * which will be prepocessed.
		 * num_feat, num_vectors are returned by reference
		 * caller has to clean up
		 *
		 * possible with subset
		 *
		 * @param num_feat number of features in matrix
		 * @param num_vec number of vectors in matrix
		 * @return transposed sparse feature matrix
		 */
		SGSparseVector<ST>* get_transposed(int32_t &num_feat, int32_t &num_vec);

		/** set sparse feature matrix
		 *
		 * not possible with subset
		 *
		 * @param sm sparse feature matrix
		 *
		 */
        void set_sparse_feature_matrix(SGSparseMatrix<ST> sm);

		/** gets a copy of a full feature matrix
		 *
		 * possible with subset
		 *
		 * @return full dense feature matrix
		 */
		SGMatrix<ST> get_full_feature_matrix();

		/** creates a sparse feature matrix from a full dense feature matrix
		 * necessary to set feature_matrix, num_features and num_vectors
		 * where num_features is the column offset, and columns are linear in memory
		 * see above for definition of sparse_feature_matrix
		 *
		 * any subset is removed before
		 *
		 * @param full full feature matrix
		 */
		virtual void set_full_feature_matrix(SGMatrix<ST> full);

		/** apply preprocessor
		 *
		 * possible with subset
		 *
		 * @param force_preprocessing if preprocssing shall be forced
		 * @return if applying was successful
		 */
		virtual bool apply_preprocessor(bool force_preprocessing=false);

		/** obtain sparse features from simple features
		 *
		 * subset on input is ignored, subset of this instance is removed
		 *
		 * @param sf simple features
		 */
		void obtain_from_simple(CDenseFeatures<ST>* sf);

		/** get number of feature vectors, possibly of subset
		 *
		 * @return number of feature vectors
		 */
		virtual int32_t get_num_vectors() const;

		/** get number of features
		 *
		 * @return number of features
		 */
		int32_t get_num_features() const;

		/** set number of features
		 *
		 * Sometimes when loading sparse features not all possible dimensions
		 * are used. This may pose a problem to classifiers when being applied
		 * to higher dimensional test-data. This function allows to
		 * artificially explode the feature space
		 *
		 * @param num the number of features, must be larger
		 *        than the current number of features
		 * @return previous number of features
		 */
		int32_t set_num_features(int32_t num);

		/** get feature class
		 *
		 * @return feature class SPARSE
		 */
		virtual EFeatureClass get_feature_class() const;

		/** get feature type
		 *
		 * @return templated feature type
		 */
		virtual EFeatureType get_feature_type() const;

		/** free feature vector
		 *
		 * possible with subset
		 *
		 * @param num index of vector in cache
		 */
		void free_feature_vector(int32_t num);

		/** get number of non-zero entries in sparse feature matrix
		 *
		 * @return number of non-zero entries in sparse feature matrix
		 */
		int64_t get_num_nonzero_entries();

		/** compute a^2 on all feature vectors
		 *
		 * possible with subset
		 *
		 * @param sq the square for each vector is stored in here
		 * @return the square for each vector
		 */
		float64_t* compute_squared(float64_t* sq);

		/** compute (a-b)^2 (== a^2+b^2-2ab)
		 * usually called by kernels'/distances' compute functions
		 * works on two feature vectors, although it is a member of a single
		 * feature: can either be called by lhs or rhs.
		 *
		 * possible wiht subsets on lhs or rhs
		 *
		 * @param lhs left-hand side features
		 * @param sq_lhs squared values of left-hand side
		 * @param idx_a index of left-hand side's vector to compute
		 * @param rhs right-hand side features
		 * @param sq_rhs squared values of right-hand side
		 * @param idx_b index of right-hand side's vector to compute
		 */
		float64_t compute_squared_norm(CSparseFeatures<float64_t>* lhs,
				float64_t* sq_lhs, int32_t idx_a,
				CSparseFeatures<float64_t>* rhs, float64_t* sq_rhs,
				int32_t idx_b);

		/** load features from file
		 *
		 * any subset is removed before
		 *
		 * @param loader File object to load data from
		 */
		void load(CFile* loader);

		/** load features from file
		 *
		 * any subset is removed before
		 *
		 * @param loader File object to load data from
		 * @return label vector
		 */
		SGVector<float64_t> load_with_labels(CLibSVMFile* loader);

		/** save features to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object to write data to
		 */
		void save(CFile* writer);

		/** save features to file
		 *
		 * not possible with subset
		 *
		 * @param writer File object to write data to
		 * @param labels vector with labels to write out
		 */
		void save_with_labels(CLibSVMFile* writer, SGVector<float64_t> labels);

		/** ensure that features occur in ascending order, only call when no
		 * preprocessors are attached
		 *
		 * not possiblwe with subset
		 * */
		void sort_features();

		/** obtain the dimensionality of the feature space
		 *
		 * (not mix this up with the dimensionality of the input space, usually
		 * obtained via get_num_features())
		 *
		 * @return dimensionality
		 */
		virtual int32_t get_dim_feature_space() const;

		/** compute dot product between vector1 and vector2,
		 * appointed by their indices
		 *
		 * possible with subset of this instance and of DotFeatures
		 *
		 * @param vec_idx1 index of first vector
		 * @param df DotFeatures (of same kind) to compute dot product with
		 * @param vec_idx2 index of second vector
		 */
		virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

		/** compute dot product between vector1 and a dense vector
		 *
		 * possible with subset
		 *
		 * @param vec_idx1 index of first vector
		 * @param vec2 pointer to real valued vector
		 * @param vec2_len length of real valued vector
		 */
		virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len);

		#ifndef DOXYGEN_SHOULD_SKIP_THIS
		/** iterator for sparse features */
		struct sparse_feature_iterator
		{
			/** feature vector */
			SGSparseVector<ST> sv;

			/** vector index */
			int32_t vector_index;

			/** feature index */
			int32_t index;

			/** print details of iterator (for debugging purposes)*/
			void print_info()
			{
				SG_SPRINT("sv=%p, vidx=%d, num_feat_entries=%d, index=%d\n",
						sv.features, vector_index, sv.num_feat_entries, index);
			}
		};
		#endif

		/** iterate over the non-zero features
		 *
		 * call get_feature_iterator first, followed by get_next_feature and
		 * free_feature_iterator to cleanup
		 *
		 * possible with subset
		 *
		 * @param vector_index the index of the vector over whose components to
		 *			iterate over
		 * @return feature iterator (to be passed to get_next_feature)
		 */
		virtual void* get_feature_iterator(int32_t vector_index);

		/** iterate over the non-zero features
		 *
		 * call this function with the iterator returned by get_first_feature
		 * and call free_feature_iterator to cleanup
		 *
		 * @param index is returned by reference (-1 when not available)
		 * @param value is returned by reference
		 * @param iterator as returned by get_first_feature
		 * @return true if a new non-zero feature got returned
		 */
		virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator);

		/** clean up iterator
		 * call this function with the iterator returned by get_first_feature
		 *
		 * @param iterator as returned by get_first_feature
		 */
		virtual void free_feature_iterator(void* iterator);

		/** Creates a new CFeatures instance containing copies of the elements
		 * which are specified by the provided indices.
		 *
		 * @param indices indices of feature elements to copy
		 * @return new CFeatures instance with copies of feature data
		 */
		virtual CFeatures* copy_subset(SGVector<index_t> indices);

		/** @return object name */
		virtual const char* get_name() const { return "SparseFeatures"; }

	protected:
		/** compute feature vector for sample num
		 * if target is set the vector is written to target
		 * len is returned by reference
		 *
		 * NOT IMPLEMENTED!
		 *
		 * @param num num
		 * @param len len
		 * @param target target
		 */
		virtual SGSparseVectorEntry<ST>* compute_sparse_feature_vector(int32_t num,
			int32_t& len, SGSparseVectorEntry<ST>* target=NULL);

	private:
		void init();

	protected:

		/// array of sparse vectors of size num_vectors
		SGSparseMatrix<ST> sparse_feature_matrix;

		/** feature cache */
		CCache< SGSparseVectorEntry<ST> >* feature_cache;
};
}
#endif /* _SPARSEFEATURES__H__ */
