/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (W) 2015 Wu Lin
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#ifndef DENSESUBSAMPLESFEATURES_H
#define DENSESUBSAMPLESFEATURES_H

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{

template<class ST> class CDenseFeatures;
template<class ST> class SGVector;
class CDotFeatures;

/** SubSamples wrap CDotFeatures but only uses a subset of samples */
template<class ST> class CDenseSubSamplesFeatures: public CDotFeatures
{
public:
    /** default constructor */
	CDenseSubSamplesFeatures();

	/** constructor */
	CDenseSubSamplesFeatures(CDenseFeatures<ST> *fea, SGVector<int32_t> idx);

    /** destructor */
	virtual ~CDenseSubSamplesFeatures();

    /** get name */
    virtual const char* get_name() const { return "DenseSubSamplesFeatures"; }

	/** set the underlying features */
	void set_features(CDenseFeatures<ST> *fea);

	/** set the index into the subset of samples */
	void set_subset_idx(SGVector<int32_t> idx);

	/** duplicate feature object
	 *
	 * abstract base method
	 *
	 * @return feature object
	 */
	virtual CFeatures* duplicate() const;

	/** get feature type
	 *
	 * abstract base method
	 *
	 * @return templated feature type
	 */
	virtual EFeatureType get_feature_type() const;

	/** get feature class
	 *
	 * abstract base method
	 *
	 * @return feature class 
	 */
	virtual EFeatureClass get_feature_class() const;

	/** get number of examples/vectors, possibly corresponding to the current subset
	 *
	 * abstract base method
	 *
	 * @return number of examples/vectors/samples
	 */
	virtual int32_t get_num_vectors() const;

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality/features
	 */
	virtual int32_t get_dim_feature_space() const;

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2);

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2,
		int32_t vec2_len);

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1,
		float64_t* vec2, int32_t vec2_len, bool abs_val=false);

	/** get number of non-zero features in vector
	 *
	 * (in case accurate estimates are too expensive overestimating is OK)
	 *
	 * @param num which vector
	 * @return number of sparse features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num);

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * @param vector_index the index of the vector over whose components to
	 *			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator(int32_t vector_index);

	/** iterate over the non-zero features
	 *
	 * call this function with the iterator returned by get_feature_iterator
	 * and call free_feature_iterator to cleanup
	 *
	 * @param index is returned by reference (-1 when not available)
	 * @param value is returned by reference
	 * @param iterator as returned by get_feature_iterator
	 * @return true if a new non-zero feature got returned
	 */
	virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator);
	
	/** clean up iterator
	 * call this function with the iterator returned by get_feature_iterator
	 *
	 * @param iterator as returned by get_feature_iterator
	 */
	virtual void free_feature_iterator(void* iterator);


	/** does this class support compatible computation bewteen difference classes?
	 * for example, this->dot(rhs_prt),
	 * can rhs_prt be an instance of a difference class?
	 *
	 * @return whether this class supports compatible computation
	 */
	virtual bool support_compatible_class() const {return true;}

	/** Given a class in right hand side, does this class support compatible computation? 
	 *
	 * for example, is this->dot(rhs_prt) valid,
	 * where rhs_prt is the class in right hand side 
	 *
	 * @param rhs the class in right hand side 
	 * @return whether this class supports compatible computation
	 */
	virtual bool get_feature_class_compatibility (EFeatureClass rhs) const;
private:
	/* init */
	void init();

	/** check whether the index is out of bound
	 * @param index index of m_idx
	 */
	void check_bound(int32_t index);

	/* full samples  */
	CDenseFeatures<ST> *m_fea;

	/* the indices of subsamples of m_fea */
	SGVector<int32_t> m_idx;
};
} /*  shogun */

#endif /*  DENSESUBSAMPLESFEATURES_H */

