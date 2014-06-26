/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2010 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Written (W) 2011-2013 Heiko Strathmann
 * Copyright (C) 1999-2009 Fraunhofer Institute FIRST and Max-Planck-Society
 * Copyright (C) 2010 Berlin Institute of Technology
 */

#ifndef _DENSEFEATURES__H__
#define _DENSEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/lib/common.h>
#include <shogun/lib/Cache.h>
#include <shogun/io/File.h>
#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/lib/DataType.h>

#include <shogun/lib/SGMatrix.h>

namespace shogun {
template<class ST> class CStringFeatures;
template<class ST> class CDenseFeatures;
template<class ST> class SGMatrix;
class CDotFeatures;

/** @brief The class DenseFeatures implements dense feature matrices.
 *
 * The feature matrices are stored en-block in memory in fortran order, i.e.
 * column-by-column, where a column denotes a feature vector.
 *
 * There are get_num_vectors() many feature vectors, of dimension
 * get_num_features(). To access a feature vector call
 * get_feature_vector() and when you are done treating it call
 * free_feature_vector(). While free_feature_vector() is a NOP in most cases
 * feature vectors might have been generated on the fly (due to a number
 * preprocessors being attached to them).
 *
 * From this template class a number the following dense feature matrix types
 * are used and supported:
 *
 * \li bool matrix - CDenseFeatures<bool>
 * \li 8bit char matrix - CDenseFeatures<char>
 * \li 8bit Byte matrix - CDenseFeatures<uint8_t>
 * \li 16bit Integer matrix - CDenseFeatures<int16_t>
 * \li 16bit Word matrix - CDenseFeatures<uint16_t>
 * \li 32bit Integer matrix - CDenseFeatures<int32_t>
 * \li 32bit Unsigned Integer matrix - CDenseFeatures<uint32_t>
 * \li 32bit Float matrix - CDenseFeatures<float32_t>
 * \li 64bit Float matrix - CDenseFeatures<float64_t>
 * \li 64bit Float matrix <b>in a file</b> - CRealFileFeatures
 * \li 64bit Tangent of posterior log-odds (TOP) features from HMM - CTOPFeatures
 * \li 64bit Fisher Kernel (FK) features from HMM - CTOPFeatures
 * \li 96bit Float matrix - CDenseFeatures<floatmax_t>
 *
 * Partly) subset access is supported for this feature type.
 * Dense use the (inherited) add_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method.
 * See also CFeatures class documentation
 */
template<class ST> class CDenseFeatures: public CDotFeatures
{
public:
	/** constructor
	 *
	 * @param size cache size
	 */
	CDenseFeatures(int32_t size = 0);

	/** copy constructor */
	CDenseFeatures(const CDenseFeatures & orig);

	/** constructor
	 *
	 * @param matrix feature matrix
	 */
	CDenseFeatures(SGMatrix<ST> matrix);

	/** constructor
	 *
	 * @param src feature matrix
	 * @param num_feat number of features in matrix
	 * @param num_vec number of vectors in matrix
	 */
	CDenseFeatures(ST* src, int32_t num_feat, int32_t num_vec);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	CDenseFeatures(CFile* loader);

	/** duplicate feature object
	 *
	 * @return feature object
	 */
	virtual CFeatures* duplicate() const;

	virtual ~CDenseFeatures();

	/** free feature matrix
	 *
	 * Any subset is removed
	 */
	void free_feature_matrix();

	/** free feature matrix and cache
	 *
	 * Any subset is removed
	 */
	void free_features();

	/** get feature vector
	 * for sample num from the matrix as it is if matrix is
	 * initialized, else return preprocessed compute_feature_vector (not
	 * implemented)
	 *
	 * @param num index of feature vector
	 * @param len length is returned by reference
	 * @param dofree whether returned vector must be freed by
	 * caller via free_feature_vector
	 * @return feature vector
	 */
	ST* get_feature_vector(int32_t num, int32_t& len, bool& dofree);

	/** set feature vector num
	 *
	 * possible with subset
	 *
	 * @param vector vector
	 * @param num index if vector to set
	 */
	void set_feature_vector(SGVector<ST> vector, int32_t num);

	/** get feature vector num
	 *
	 * possible with subset
	 *
	 * @param num index of vector
	 * @return feature vector
	 */
	SGVector<ST> get_feature_vector(int32_t num);

	/** free feature vector
	 *
	 * possible with subset
	 *
	 * @param feat_vec feature vector to free
	 * @param num index in feature cache
	 * @param dofree if vector should be really deleted
	 */
	void free_feature_vector(ST* feat_vec, int32_t num, bool dofree);

	/** free feature vector
	 *
	 * possible with subset
	 *
	 * @param vec feature vector to free
	 * @param num index in feature cache
	 */
	void free_feature_vector(SGVector<ST> vec, int32_t num);

	/**
	 * Extracts the feature vectors mentioned in idx and replaces them in
	 * feature matrix in place.
	 *
	 * It does not resize the allocated memory block.
	 *
	 * not possible with subset
	 *
	 * @param idx index with examples that shall remain in the feature matrix
	 * @param idx_len length of the index
	 *
	 * Note: assumes idx is sorted
	 */
	void vector_subset(int32_t* idx, int32_t idx_len);

	/**
	 * Extracts the features mentioned in idx and replaces them in
	 * feature matrix in place.
	 *
	 * It does not resize the allocated memory block.
	 *
	 * Not possible with subset.
	 *
	 * @param idx index with features that shall remain in the feature matrix
	 * @param idx_len length of the index
	 *
	 * Note: assumes idx is sorted
	 */
	void feature_subset(int32_t* idx, int32_t idx_len);

	/** Getter the feature matrix
	 *
	 * in-place without subset
	 * a copy with subset
	 *
	 * @return matrix feature matrix
	 */
	SGMatrix<ST> get_feature_matrix();

	/** steals feature matrix, i.e. returns matrix and
	 * forget about it
	 * subset is ignored
	 *
	 * @return matrix feature matrix
	 */
	SGMatrix<ST> steal_feature_matrix();

	/** Setter for feature matrix
	 *
	 * any subset is removed
	 *
	 * num_cols is number of feature vectors
	 * num_rows is number of dims of vectors
	 * see below for definition of feature_matrix
	 *
	 * @param matrix feature matrix to set
	 *
	 */
	void set_feature_matrix(SGMatrix<ST> matrix);

	/** get the pointer to the feature matrix
	 * num_feat,num_vectors are returned by reference
	 *
	 * subset is ignored
	 *
	 * @param num_feat number of features in matrix
	 * @param num_vec number of vectors in matrix
	 * @return feature matrix
	 */
	ST* get_feature_matrix(int32_t &num_feat, int32_t &num_vec);

	/** get a transposed copy of the features
	 *
	 * possible with subset
	 *
	 * @return transposed copy
	 */
	CDenseFeatures<ST>* get_transposed();

	/** compute and return the transpose of the feature matrix
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
	ST* get_transposed(int32_t &num_feat, int32_t &num_vec);

	/** copy feature matrix
	 * store copy of feature_matrix, where num_features is the
	 * column offset, and columns are linear in memory
	 * see below for definition of feature_matrix
	 *
	 * not possible with subset
	 *
	 * @param src feature matrix to copy
	 */
	virtual void copy_feature_matrix(SGMatrix<ST> src);

	/** obtain dense features from other dotfeatures
	 *
	 * removes any subset before
	 *
	 * @param df dotfeatures to obtain features from
	 */
	void obtain_from_dot(CDotFeatures* df);

	/** apply preprocessor
	 *
	 * applies preprocessors to ALL features (subset removed before and
	 * restored afterwards)
	 *
	 * not possible with subset
	 *
	 * @param force_preprocessing if preprocssing shall be forced
	 * @return if applying was successful
	 */
	virtual bool apply_preprocessor(bool force_preprocessing = false);

	/** get number of feature vectors
	 *
	 * @return number of feature vectors
	 */
	virtual int32_t get_num_vectors() const;

	/** get number of features (of possible subset)
	 *
	 * @return number of features
	 */
	int32_t get_num_features() const;

	/** set number of features
	 *
	 * @param num number to set
	 */
	void set_num_features(int32_t num);

	/** set number of vectors
	 *
	 * not possible with subset
	 *
	 * @param num number to set
	 */
	void set_num_vectors(int32_t num);

	/** Initialize cache
	 *
	 * not possible with subset
	 */
	void initialize_cache();

	/** get feature class
	 *
	 * @return feature class DENSE
	 */
	virtual EFeatureClass get_feature_class() const;

	/** get feature type
	 *
	 * @return templated feature type
	 */
	virtual EFeatureType get_feature_type() const;

	/** reshape
	 *
	 * not possible with subset
	 *
	 * @param p_num_features new number of features
	 * @param p_num_vectors new number of vectors
	 * @return if reshaping was successful
	 */
	virtual bool reshape(int32_t p_num_features, int32_t p_num_vectors);

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
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df,
			int32_t vec_idx2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2,
			int32_t vec2_len);

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * possible with subset
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
			float64_t* vec2, int32_t vec2_len, bool abs_val = false);

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num);

	/** load features from file
	 *
	 * @param loader File object via which to load data
	 */
	virtual void load(CFile* loader);

	/** save features to file
	 *
	 * @param saver File object via which to save data
	 */
	virtual void save(CFile* saver);

#ifndef DOXYGEN_SHOULD_SKIP_THIS
	/** iterator for dense features */
	struct dense_feature_iterator
	{
		/** pointer to feature vector */
		ST* vec;
		/** index of vector */
		int32_t vidx;
		/** length of vector */
		int32_t vlen;
		/** if we need to free the vector*/
		bool vfree;

		/** feature index */
		int32_t index;
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
	 * possible with subset
	 *
	 * @param index is returned by reference (-1 when not available)
	 * @param value is returned by reference
	 * @param iterator as returned by get_first_feature
	 * @return true if a new non-zero feature got returned
	 */
	virtual bool get_next_feature(int32_t& index, float64_t& value,
			void* iterator);

	/** clean up iterator
	 * call this function with the iterator returned by get_first_feature
	 *
	 * @param iterator as returned by get_first_feature
	 */
	virtual void free_feature_iterator(void* iterator);

	/** Creates a new CFeatures instance containing copies of the elements
	 * which are specified by the provided indices.
	 *
	 * possible with subset
	 *
	 * @param indices indices of feature elements to copy
	 * @return new CFeatures instance with copies of feature data
	 */
	virtual CFeatures* copy_subset(SGVector<index_t> indices);

	/** Creates a new CFeatures instance containing only the dimensions
	 * of the feature vector which are specified by the provided indices.
	 *
	 * This method is needed for feature selection tasks
	 *
	 * possible with subset
	 *
	 * @param dims indices of feature dimensions to copy
	 * @return new CFeatures instance with copies of specified features
	 */
	virtual CFeatures* copy_dimension_subset(SGVector<index_t> dims);

	/** checks if the contents of this CDenseFeatures object are the same to
	 * the contents of rhs
	 *
	 * @param rhs other CDenseFeatures object to compare to this one
	 * @return whether they represent the same information
	 */
	virtual bool is_equal(CDenseFeatures* rhs);

	/** Takes a list of feature instances and returns a new instance which is
	 * a concatenation of a copy if this instace's data and the given
	 * instancess data. Note that the feature types have to be equal.
	 *
	 * @param other feature object to append
	 * @return new feature object which contains copy of data of this
	 * instance and of given one
	 */
	CFeatures* create_merged_copy(CList* other);

	/** Convenience method for method with same name and list as parameter.
	 *
	 * @param other feature object to append
	 * @return new feature object which contains copy of data of this
	 * instance and of given one
	 */
	CFeatures* create_merged_copy(CFeatures* other);

	/** helper method used to specialize a base class instance
	 *
	 */
	static CDenseFeatures* obtain_from_generic(CFeatures* const base_features);

	/** @return object name */
	virtual const char* get_name() const { return "DenseFeatures"; }

protected:
	/** compute feature vector for sample num
	 * if target is set the vector is written to target
	 * len is returned by reference
	 *
	 * NOT IMPLEMENTED!
	 *
	 * @param num num
	 * @param len len
	 * @param target
	 * @return feature vector
	 */
	virtual ST* compute_feature_vector(int32_t num, int32_t& len,
			ST* target = NULL);

private:
	void init();

protected:
	/// number of vectors in cache
	int32_t num_vectors;

	/// number of features in cache
	int32_t num_features;

	/** Feature matrix and its associated number of
	 * vectors and features. Note that num_vectors / num_features
	 * above match matrix sizes if feature_matrix.matrix != NULL
	 * */
	SGMatrix<ST> feature_matrix;

	/** feature cache */
	CCache<ST>* feature_cache;
};
}
#endif // _DENSEFEATURES__H__
