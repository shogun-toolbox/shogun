/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#ifndef DENSESUBSETFEATURES_H__
#define DENSESUBSETFEATURES_H__

#include <shogun/lib/config.h>

#include <shogun/features/DenseFeatures.h>
#include <shogun/features/DotFeatures.h>

namespace shogun
{

template<class ST> class CDenseFeatures;
template<class ST> class SGVector;
class CDotFeatures;

/** SubsetFeatures wrap features but only uses a subset of the variables */
template<class ST> class CDenseSubsetFeatures: public CDotFeatures
{
public:
    /** default constructor */
	CDenseSubsetFeatures():m_fea(NULL) { set_generic<ST>(); }

	/** constructor */
	CDenseSubsetFeatures(CDenseFeatures<ST> *fea, SGVector<int32_t> idx)
		:m_fea(fea), m_idx(idx) { SG_REF(m_fea); set_generic<ST>(); }

    /** destructor */
	virtual ~CDenseSubsetFeatures() { SG_UNREF(m_fea); }

    /** get name */
    virtual const char* get_name() const { return "DenseSubsetFeatures"; }

	/** set the underlying features */
	void set_features(CDenseFeatures<ST> *fea)
	{
		SG_REF(fea);
		SG_UNREF(m_fea);
		m_fea = fea;
	}

	/** set the index into the subset of features */
	void set_subset_idx(SGVector<int32_t> idx)
	{
		m_idx = idx;
	}

	/** duplicate feature object
	 *
	 * abstract base method
	 *
	 * @return feature object
	 */
	virtual CFeatures* duplicate() const
	{
		return new CDenseSubsetFeatures(m_fea, m_idx);
	}

	/** get feature type
	 *
	 * abstract base method
	 *
	 * @return templated feature type
	 */
	virtual EFeatureType get_feature_type() const
	{
		return m_fea->get_feature_type();
	}

	/** get feature class
	 *
	 * abstract base method
	 *
	 * @return feature class like STRING, SIMPLE, SPARSE...
	 */
	virtual EFeatureClass get_feature_class() const
	{
		return m_fea->get_feature_class();
	}

	/** get number of examples/vectors, possibly corresponding to the current subset
	 *
	 * abstract base method
	 *
	 * @return number of examples/vectors (possibly of subset, if implemented)
	 */
	virtual int32_t get_num_vectors() const
	{
		return m_fea->get_num_vectors();
	}

	/** obtain the dimensionality of the feature space
	 *
	 * (not mix this up with the dimensionality of the input space, usually
	 * obtained via get_num_features())
	 *
	 * @return dimensionality
	 */
	virtual int32_t get_dim_feature_space() const
	{
		return m_idx.vlen;
	}

	/** compute dot product between vector1 and vector2,
	 * appointed by their indices
	 *
	 * @param vec_idx1 index of first vector
	 * @param df DotFeatures (of same kind) to compute dot product with
	 * @param vec_idx2 index of second vector
	 */
	virtual float64_t dot(int32_t vec_idx1, CDotFeatures* df, int32_t vec_idx2)
	{
		CDenseSubsetFeatures<ST> *dsf = dynamic_cast<CDenseSubsetFeatures<ST> *>(df);
		if (dsf == NULL)
			SG_ERROR("Require DenseSubsetFeatures of the same kind to perform dot\n")

		if (m_idx.vlen != dsf->m_idx.vlen)
			SG_ERROR("Cannot dot vectors of different length\n")

		SGVector<ST> vec1 = m_fea->get_feature_vector(vec_idx1);
		SGVector<ST> vec2 = dsf->m_fea->get_feature_vector(vec_idx2);

		float64_t sum = 0;
		for (int32_t i=0; i < m_idx.vlen; ++i)
			sum += vec1[m_idx[i]] * vec2[dsf->m_idx[i]];

		return sum;
	}

	/** compute dot product between vector1 and a dense vector
	 *
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 */
	virtual float64_t dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len)
	{
		if (m_idx.vlen != vec2_len)
			SG_ERROR("Cannot dot vectors of different length\n")
		SGVector<ST> vec1 = m_fea->get_feature_vector(vec_idx1);

		float64_t sum=0;
		for (int32_t i=0; i < vec2_len; ++i)
			sum += vec1[m_idx[i]] * vec2[i];

		return sum;
	}

	/** add vector 1 multiplied with alpha to dense vector2
	 *
	 * @param alpha scalar alpha
	 * @param vec_idx1 index of first vector
	 * @param vec2 pointer to real valued vector
	 * @param vec2_len length of real valued vector
	 * @param abs_val if true add the absolute value
	 */
	virtual void add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val=false)
	{
		if (m_idx.vlen != vec2_len)
			SG_ERROR("Cannot add_to_dense_vec vectors of different length\n")

		SGVector<ST> vec1 = m_fea->get_feature_vector(vec_idx1);
		if (abs_val)
		{
			for (int32_t i=0; i < vec2_len; ++i)
				vec2[i] += alpha * CMath::abs(vec1[m_idx[i]]);
		}
		else
		{
			for (int32_t i=0; i < vec2_len; ++i)
				vec2[i] += alpha * vec1[m_idx[i]];
		}
	}

	/** get number of non-zero features in vector
	 *
	 * (in case accurate estimates are too expensive overestimating is OK)
	 *
	 * @param num which vector
	 * @return number of sparse features in vector
	 */
	virtual int32_t get_nnz_features_for_vector(int32_t num)
	{
		return m_idx.vlen;
	}

	/** iterate over the non-zero features
	 *
	 * call get_feature_iterator first, followed by get_next_feature and
	 * free_feature_iterator to cleanup
	 *
	 * @param vector_index the index of the vector over whose components to
	 *			iterate over
	 * @return feature iterator (to be passed to get_next_feature)
	 */
	virtual void* get_feature_iterator(int32_t vector_index)
	{
		SG_NOTIMPLEMENTED
		return NULL;
	}

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
	virtual bool get_next_feature(int32_t& index, float64_t& value, void* iterator)
	{
		SG_NOTIMPLEMENTED
		return false;
	}

	/** clean up iterator
	 * call this function with the iterator returned by get_feature_iterator
	 *
	 * @param iterator as returned by get_feature_iterator
	 */
	virtual void free_feature_iterator(void* iterator)
	{
		SG_NOTIMPLEMENTED
	}
private:
	CDenseFeatures<ST> *m_fea;
	SGVector<int32_t> m_idx;
};
} /*  shogun */

#endif /* end of include guard: DENSESUBSETFEATURES_H__ */

