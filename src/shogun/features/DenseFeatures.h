/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soumyajit De, Heiko Strathmann, Saurabh Mahindre, Soeren Sonnenburg,
 *          Vladislav Horbatiuk, Yuyu Zhang, Kevin Hughes, Evgeniy Andreev,
 *          Thoralf Klein, Fernando Iglesias, Bjoern Esser, Sergey Lisitsyn
 */

#ifndef _DENSEFEATURES__H__
#define _DENSEFEATURES__H__

#include <shogun/lib/config.h>

#include <shogun/features/DotFeatures.h>
#include <shogun/features/StringFeatures.h>
#include <shogun/io/File.h>
#include <shogun/lib/Cache.h>
#include <shogun/lib/DataType.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/common.h>

namespace shogun {
template<class ST> class StringFeatures;
template<class ST> class DenseFeatures;
template<class ST> class SGMatrix;
class DotFeatures;

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
 * \li bool matrix - DenseFeatures<bool>
 * \li 8bit char matrix - DenseFeatures<char>
 * \li 8bit Byte matrix - DenseFeatures<uint8_t>
 * \li 16bit Integer matrix - DenseFeatures<int16_t>
 * \li 16bit Word matrix - DenseFeatures<uint16_t>
 * \li 32bit Integer matrix - DenseFeatures<int32_t>
 * \li 32bit Unsigned Integer matrix - DenseFeatures<uint32_t>
 * \li 32bit Float matrix - DenseFeatures<float32_t>
 * \li 64bit Float matrix - DenseFeatures<float64_t>
 * \li 64bit Float matrix <b>in a file</b> - CRealFileFeatures
 * \li 64bit Tangent of posterior log-odds (TOP) features from HMM - CTOPFeatures
 * \li 64bit Fisher Kernel (FK) features from HMM - CTOPFeatures
 * \li 96bit Float matrix - DenseFeatures<floatmax_t>
 *
 * Partly) subset access is supported for this feature type.
 * Dense use the (inherited) add_subset(), remove_subset() functions.
 * If done, all calls that work with features are translated to the subset.
 * See comments to find out whether it is supported for that method.
 * See also Features class documentation
 */
template<class ST> class DenseFeatures: public DotFeatures
{
public:
	/** constructor
	 *
	 * @param size cache size
	 */
	DenseFeatures(int32_t size = 0);

	/** copy constructor */
	DenseFeatures(const DenseFeatures & orig);

	/** constructor
	 *
	 * @param matrix feature matrix
	 */
	DenseFeatures(SGMatrix<ST> matrix);

	/** constructor
	 *
	 * @param src feature matrix
	 * @param num_feat number of features in matrix
	 * @param num_vec number of vectors in matrix
	 */
	DenseFeatures(ST* src, int32_t num_feat, int32_t num_vec);

	/** constructor from DotFeatures
	 *
	 * @param features DotFeatures object
	 */
	DenseFeatures(std::shared_ptr<DotFeatures> features);

	/** constructor loading features from file
	 *
	 * @param loader File object via which to load data
	 */
	DenseFeatures(std::shared_ptr<File> loader);

	/** duplicate feature object
	 *
	 * @return feature object
	 */
	virtual std::shared_ptr<Features> duplicate() const;

	virtual ~DenseFeatures();

	/** free feature matrix
	 *
	 * Any subset is removed
	 */
	void free_feature_matrix();

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
	ST* get_feature_vector(int32_t num, int32_t& len, bool& dofree) const;

	/** get feature vector num
	 *
	 * possible with subset
	 *
	 * @param num index of vector
	 * @return feature vector
	 */
	SGVector<ST> get_feature_vector(int32_t num) const;

	/** free feature vector
	 *
	 * possible with subset
	 *
	 * @param feat_vec feature vector to free
	 * @param num index in feature cache
	 * @param dofree if vector should be really deleted
	 */
	void free_feature_vector(ST* feat_vec, int32_t num, bool dofree) const;

	/** free feature vector
	 *
	 * possible with subset
	 *
	 * @param vec feature vector to free
	 * @param num index in feature cache
	 */
	void free_feature_vector(SGVector<ST> vec, int32_t num) const;

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
	SGMatrix<ST> get_feature_matrix() const;

	/** get the pointer to the feature matrix
	 * num_feat,num_vectors are returned by reference
	 *
	 * subset is ignored
	 *
	 * @param num_feat number of features in matrix
	 * @param num_vec number of vectors in matrix
	 * @return feature matrix
	 */
	ST* get_feature_matrix(int32_t& num_feat, int32_t& num_vec) const;

	/** get a transposed copy of the features
	 *
	 * possible with subset
	 *
	 * @return transposed copy
	 */
	std::shared_ptr<DenseFeatures<ST>> get_transposed();

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
	virtual float64_t dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df,
			int32_t vec_idx2) const;

	/** Computes the sum of all feature vectors
	 * @return Sum of all feature vectors
	 */
	SGVector<ST> sum() const;

	/** Computes the empirical mean of all feature vectors
	 * @return Mean of all feature vectors
	 */
	SGVector<ST> mean() const;

	/** Computes the standard deviation of all feature vectors
	 * @param colwise if true calculates feature wise standard deviation,
	 * otherwise calculates the matrix standard deviation
	 * @return Standard deviation of all feature vectors or of whole matrix
	 */
	SGVector<float64_t > std(bool colwise = true) const;

	/** Computes the \f$DxD\f$ (uncentered, un-normalized) covariance matrix
	 *
	 *\f[
	 * X X^\top
	 * \f]
	 *
	 * where \f$X\f$ is the \f$DxN\f$ dimensional feature matrix with \f$N\f$
	 * feature vectors of dimension \f$D\f$.
	 */
	SGMatrix<ST> cov() const;
	/** Computes the \f$fNxN\f$ (uncentered, un-normalized) gram matrix of
	 * pairwise dot products, that is
	 *
	 *\f[
	 * X^\top X
	 * \f]
	 *
	 * where \f$X\f$ is the \f$DxN\f$ dimensional feature matrix with \f$N\f$
	 * feature vectors of dimension \f$D\f$.
	 */
	SGMatrix<ST> gram() const;

	/** Computes the dot product of the feature matrix with a given vector.
	 *
	 * @param other Vector to compute dot products with, size must match number
	 * of feature vectors
	 * @return Vector as many entries as feature dimensions
	 */
	SGVector<ST> dot(const SGVector<ST>& other) const;

	/** compute dot product between vector1 and a dense vector
	 *
	 * possible with subset
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2,
			int32_t vec2_len) const;

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
			float64_t* vec2, int32_t vec2_len, bool abs_val = false) const;

	/** get number of non-zero features in vector
	 *
	 * @param num which vector
	 * @return number of non-zero features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num) const;

	/** load features from file
	 *
	 * @param loader File object via which to load data
	 */
	virtual void load(std::shared_ptr<File> loader);

	/** save features to file
	 *
	 * @param saver File object via which to save data
	 */
	virtual void save(std::shared_ptr<File> saver);

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

	/** Creates a new Features instance containing copies of the elements
	 * which are specified by the provided indices.
	 *
	 * possible with subset
	 *
	 * @param indices indices of feature elements to copy
	 * @return new Features instance with copies of feature data
	 */
	virtual std::shared_ptr<Features> copy_subset(SGVector<index_t> indices) const;

	/** Creates a new Features instance containing only the dimensions
	 * of the feature vector which are specified by the provided indices.
	 *
	 * This method is needed for feature selection tasks
	 *
	 * possible with subset
	 *
	 * @param dims indices of feature dimensions to copy
	 * @return new Features instance with copies of specified features
	 */
	virtual std::shared_ptr<Features> copy_dimension_subset(SGVector<index_t> dims) const;

	/** checks if the contents of this DenseFeatures object are the same to
	 * the contents of rhs
	 *
	 * @param rhs other DenseFeatures object to compare to this one
	 * @return whether they represent the same information
	 */
	virtual bool is_equal(std::shared_ptr<DenseFeatures> rhs);

	/** Takes a list of feature instances and returns a new instance which is
	 * a concatenation of a copy if this instace's data and the given
	 * instancess data. Note that the feature types have to be equal.
	 * This method respects the subsets for all the feature instances involved.
	 *
	 * @param other feature object to append
	 * @return new feature object which contains copy of data of this
	 * instance and of given one
	 */
	std::shared_ptr<Features> create_merged_copy(const std::vector<std::shared_ptr<Features>>& other) const override;

	/** Convenience method for method with same name and list as parameter.
	 *
	 * @param other feature object to append
	 * @return new feature object which contains copy of data of this
	 * instance and of given one
	 */
	std::shared_ptr<Features> create_merged_copy(std::shared_ptr<Features> other) const;

/** helper method used to specialize a base class instance
 *
 */
#ifndef SWIG
	[[deprecated("use .as template function")]]
#endif
	static std::shared_ptr<DenseFeatures> obtain_from_generic(std::shared_ptr<Features> base_features);

#ifndef SWIG // SWIG should skip this part
	virtual std::shared_ptr<Features> shallow_subset_copy();
#endif

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
			ST* target = NULL) const;

	/** free feature matrix and cache
	 *
	 * Any subset is removed
	 */
	void free_features();

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

private:
	void init();

protected:
	/*
	 * Helper method which copies the working feature matrix into the pre-allocated
	 * target matrix passed to this method. If the size of the pre-allocated matrix
	 * is not sufficient to copy all the feature vectors, considering the column
	 * offset, it throws an error. It then copies into the target matrix, starting
	 * from base + (colum_offset * num_features) location.
	 */
	void copy_feature_matrix(SGMatrix<ST> target, index_t column_offset=0) const;

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
	std::shared_ptr<Cache<ST>> feature_cache;
};
}
#endif // _DENSEFEATURES__H__
