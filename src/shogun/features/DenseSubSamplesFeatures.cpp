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
#include <shogun/features/DenseSubSamplesFeatures.h>

namespace shogun
{

template<class ST> DenseSubSamplesFeatures<ST>::DenseSubSamplesFeatures()
	:DotFeatures()
{
	init();
}

template<class ST> DenseSubSamplesFeatures<ST>::DenseSubSamplesFeatures(
	std::shared_ptr<DenseFeatures<ST>> fea, SGVector<int32_t> idx)
	:DotFeatures()
{
	init();
	set_features(fea);
	set_subset_idx(idx);
}

template<class ST> DenseSubSamplesFeatures<ST>::~DenseSubSamplesFeatures()
{

}

template<class ST> void DenseSubSamplesFeatures<ST>::set_features(std::shared_ptr<DenseFeatures<ST>> fea)
{
	if (m_fea.get()!=fea.get())
	{
		m_fea = fea;
	}
}

template<class ST> void DenseSubSamplesFeatures<ST>::init()
{
	set_generic<ST>();
	m_fea=NULL;
	m_idx=SGVector<int32_t>();
	SG_ADD(&m_idx, "idx", "idx");
	SG_ADD((std::shared_ptr<SGObject>*)&m_fea, "fea", "fea");
}

template<class ST> std::shared_ptr<Features> DenseSubSamplesFeatures<ST>::duplicate() const
{
	return std::make_shared<DenseSubSamplesFeatures>(m_fea, m_idx);
}

template<class ST> bool DenseSubSamplesFeatures<ST>::get_feature_class_compatibility (EFeatureClass rhs) const
{
	if (this->get_feature_class()==rhs || m_fea->get_feature_class()==rhs)
		return true;
	return false;
}

template<class ST> EFeatureType DenseSubSamplesFeatures<ST>::get_feature_type() const

{
	return m_fea->get_feature_type();
}

template<class ST> EFeatureClass DenseSubSamplesFeatures<ST>::get_feature_class() const
{
	return C_SUB_SAMPLES_DENSE;
}

template<class ST> int32_t DenseSubSamplesFeatures<ST>::get_dim_feature_space() const
{
	return m_fea->get_dim_feature_space();
}

template<class ST> int32_t DenseSubSamplesFeatures<ST>::get_num_vectors() const
{
	return m_idx.vlen;
}

template<class ST> void DenseSubSamplesFeatures<ST>::set_subset_idx(SGVector<int32_t> idx)
{
	REQUIRE(m_fea, "Please set the features first\n");
	int32_t total_vlen=m_fea->get_num_vectors();
	for (int32_t i=0; i<idx.vlen; i++)
	{
		int32_t index=idx[i];
		REQUIRE(index>=0 && index<total_vlen,
			"The index (%d) in the vector idx is not valid since there are %d samples\n",
			index, total_vlen);
	}
	m_idx = idx;
}

template<class ST> float64_t DenseSubSamplesFeatures<ST>::dot(
	int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	check_bound(vec_idx1);
	auto df_f= std::dynamic_pointer_cast<DenseSubSamplesFeatures<ST>>(df);
	float64_t res=0.0;
	if (df_f)
	{
		REQUIRE(df_f->get_feature_type()==get_feature_type(),"Features type do not match\n");
		REQUIRE(df_f->get_feature_class()==get_feature_class(),"Features class do not match\n");
		df_f->check_bound(vec_idx2);
		res=m_fea->dot(m_idx[vec_idx1], df_f->m_fea, df_f->m_idx[vec_idx2]);
	}
	else
		res=m_fea->dot(m_idx[vec_idx1], df, vec_idx2);
	return res;
}

template<class ST> float64_t DenseSubSamplesFeatures<ST>::dense_dot(
	int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len) const
{
	check_bound(vec_idx1);
	return m_fea->dense_dot(m_idx[vec_idx1], vec2, vec2_len);
}

template<class ST> void DenseSubSamplesFeatures<ST>::check_bound(int32_t index) const
{
	REQUIRE(index<m_idx.vlen && index>=0,
		"Index (%d) is out of bound (%d)\n", index, m_idx.vlen);
}

template<class ST> void DenseSubSamplesFeatures<ST>::add_to_dense_vec(
	float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	check_bound(vec_idx1);
	m_fea->add_to_dense_vec(alpha, m_idx[vec_idx1], vec2, vec2_len, abs_val);
}

template<class ST> int32_t DenseSubSamplesFeatures<ST>::get_nnz_features_for_vector(
	int32_t num) const
{
	return m_fea->get_nnz_features_for_vector(num);
}

template<class ST> void* DenseSubSamplesFeatures<ST>::get_feature_iterator(
	int32_t vector_index)
{
	SG_NOTIMPLEMENTED;
	return NULL;
}

template<class ST> bool DenseSubSamplesFeatures<ST>::get_next_feature(
	int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED;
	return false;
}

template<class ST> void DenseSubSamplesFeatures<ST>::free_feature_iterator(
	void* iterator)
{
	SG_NOTIMPLEMENTED
}

template class DenseSubSamplesFeatures<bool>;
template class DenseSubSamplesFeatures<char>;
template class DenseSubSamplesFeatures<int8_t>;
template class DenseSubSamplesFeatures<uint8_t>;
template class DenseSubSamplesFeatures<int16_t>;
template class DenseSubSamplesFeatures<uint16_t>;
template class DenseSubSamplesFeatures<int32_t>;
template class DenseSubSamplesFeatures<uint32_t>;
template class DenseSubSamplesFeatures<int64_t>;
template class DenseSubSamplesFeatures<uint64_t>;
template class DenseSubSamplesFeatures<float32_t>;
template class DenseSubSamplesFeatures<float64_t>;
template class DenseSubSamplesFeatures<floatmax_t>;

}
