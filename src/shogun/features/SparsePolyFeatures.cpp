/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Vladislav Horbatiuk, Jacob Walker,
 *          Evgeniy Andreev, Viktor Gal, Evan Shelhamer, Sergey Lisitsyn,
 *          Evangelos Anagnostopoulos, Heiko Strathmann
 */
#include <shogun/features/SparsePolyFeatures.h>
#include <shogun/lib/Hash.h>

using namespace shogun;

SparsePolyFeatures::SparsePolyFeatures(): DotFeatures()
{
	SG_UNSTABLE("CSparsePolyFeatures::CSparsePolyFeatures()",
				"\n");

	m_feat = NULL;
	m_degree = 0;
	m_normalize = false;
	m_input_dimensions = 0;
	m_output_dimensions = 0;
	m_normalization_values = NULL;
	mask = 0;
	m_hash_bits = 0;
}

SparsePolyFeatures::SparsePolyFeatures(std::shared_ptr<SparseFeatures<float64_t>> feat, int32_t degree, bool normalize, int32_t hash_bits)
	: DotFeatures(), m_normalization_values(NULL)
{
	ASSERT(feat)

	m_feat = feat;

	m_degree=degree;
	m_normalize=normalize;
	m_hash_bits=hash_bits;
	mask=(uint32_t) (((uint64_t) 1)<<m_hash_bits)-1;
	m_output_dimensions=1<<m_hash_bits;
	m_input_dimensions=feat->get_num_features();

	if (m_normalize)
		store_normalization_values();
}

SparsePolyFeatures::~SparsePolyFeatures()
{
	SG_FREE(m_normalization_values);

}

SparsePolyFeatures::SparsePolyFeatures(const SparsePolyFeatures & orig)
{
	SG_PRINT("CSparsePolyFeatures:\n")
	SG_NOTIMPLEMENTED
}

int32_t SparsePolyFeatures::get_dim_feature_space() const
{
	return m_output_dimensions;
}

int32_t SparsePolyFeatures::get_nnz_features_for_vector(int32_t num) const
{
	int32_t vlen;
	SGSparseVector<float64_t> vec=m_feat->get_sparse_feature_vector(num);
	vlen=vec.num_feat_entries;
	m_feat->free_feature_vector(num);
	return vlen*(vlen+1)/2;
}

EFeatureType SparsePolyFeatures::get_feature_type() const
{
	return F_UNKNOWN;
}

EFeatureClass SparsePolyFeatures::get_feature_class() const
{
	return C_POLY;
}

int32_t SparsePolyFeatures::get_num_vectors() const
{
	if (m_feat)
		return m_feat->get_num_vectors();
	else
		return 0;

}

void* SparsePolyFeatures::get_feature_iterator(int32_t vector_index)
{
	SG_NOTIMPLEMENTED
	return NULL;
}

bool SparsePolyFeatures::get_next_feature(int32_t& index, float64_t& value, void* iterator)
{
	SG_NOTIMPLEMENTED
	return false;
}

void SparsePolyFeatures::free_feature_iterator(void* iterator)
{
	SG_NOTIMPLEMENTED
}

float64_t SparsePolyFeatures::dot(int32_t vec_idx1, std::shared_ptr<DotFeatures> df, int32_t vec_idx2) const
{
	ASSERT(df)
	ASSERT(df->get_feature_type() == get_feature_type())
	ASSERT(df->get_feature_class() == get_feature_class())

	auto pf=std::static_pointer_cast<SparsePolyFeatures>(df);

	SGSparseVector<float64_t> vec1=m_feat->get_sparse_feature_vector(vec_idx1);
	SGSparseVector<float64_t> vec2=pf->m_feat->get_sparse_feature_vector(
			vec_idx2);

	float64_t result=SGSparseVector<float64_t>::sparse_dot(vec1, vec2);
	result=Math::pow(result, m_degree);

	m_feat->free_feature_vector(vec_idx1);
	pf->m_feat->free_feature_vector(vec_idx2);

	return result;
}

float64_t SparsePolyFeatures::dense_dot(int32_t vec_idx1, const float64_t* vec2, int32_t vec2_len) const
{
	if (vec2_len != m_output_dimensions)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, m_output_dimensions=%d\n", vec2_len, m_output_dimensions)

	SGSparseVector<float64_t> vec=m_feat->get_sparse_feature_vector(vec_idx1);

	float64_t result=0;

	if (vec.features)
	{
		if (m_degree==2)
		{
			/* (a+b)^2 = a^2 + 2ab +b^2 */
			for (int32_t i=0; i<vec.num_feat_entries; i++)
			{
				float64_t v1=vec.features[i].entry;
				uint32_t seed=Hash::MurmurHash3(
						(uint8_t*)&(vec.features[i].feat_index),
						sizeof(int32_t), 0xDEADBEAF);

				for (int32_t j=i; j<vec.num_feat_entries; j++)
				{
					float64_t v2=vec.features[j].entry;
					uint32_t h=Hash::MurmurHash3(
							(uint8_t*)&(vec.features[j].feat_index),
							sizeof(int32_t), seed)&mask;
					float64_t v;

					if (i==j)
						v=v1*v1;
					else
						v = std::sqrt(2.0) * v1 * v2;

					result+=v*vec2[h];
				}
			}
		}
		else if (m_degree==3)
			SG_NOTIMPLEMENTED
	}

	if (m_normalize)
		result/=m_normalization_values[vec_idx1];

	m_feat->free_feature_vector(vec_idx1);
	return result;
}

void SparsePolyFeatures::add_to_dense_vec(float64_t alpha, int32_t vec_idx1, float64_t* vec2, int32_t vec2_len, bool abs_val) const
{
	if (vec2_len!=m_output_dimensions)
		SG_ERROR("Dimensions don't match, vec2_dim=%d, m_output_dimensions=%d\n", vec2_len, m_output_dimensions)

	SGSparseVector<float64_t> vec=m_feat->get_sparse_feature_vector(vec_idx1);

	float64_t norm_val=1.0;
	if (m_normalize)
		norm_val = m_normalization_values[vec_idx1];
	alpha/=norm_val;

	if (m_degree==2)
	{
		/* (a+b)^2 = a^2 + 2ab +b^2 */
		for (int32_t i=0; i<vec.num_feat_entries; i++)
		{
			float64_t v1=vec.features[i].entry;
			uint32_t seed=Hash::MurmurHash3(
					(uint8_t*)&(vec.features[i].feat_index), sizeof(int32_t),
					0xDEADBEAF);

			for (int32_t j=i; j<vec.num_feat_entries; j++)
			{
				float64_t v2=vec.features[j].entry;
				uint32_t h=Hash::MurmurHash3(
						(uint8_t*)&(vec.features[j].feat_index),
						sizeof(int32_t), seed)&mask;
				float64_t v;

				if (i==j)
					v=alpha*v1*v1;
				else
					v = alpha * std::sqrt(2.0) * v1 * v2;

				if (abs_val)
					vec2[h]+=Math::abs(v);
				else
					vec2[h]+=v;
			}
		}
	}
	else if (m_degree==3)
		SG_NOTIMPLEMENTED

	m_feat->free_feature_vector(vec_idx1);
}

void SparsePolyFeatures::store_normalization_values()
{
	SG_FREE(m_normalization_values);

	m_normalization_values_len = this->get_num_vectors();

	m_normalization_values=SG_MALLOC(float64_t, m_normalization_values_len);
	for (int i=0; i<m_normalization_values_len; i++)
	{
		float64_t val = std::sqrt(dot(i,
			std::dynamic_pointer_cast<std::remove_pointer_t<decltype(this)>>(shared_from_this()), i));
		if (val==0)
			// trap division by zero
			m_normalization_values[i]=1.0;
		else
			m_normalization_values[i]=val;
	}

}

std::shared_ptr<Features> SparsePolyFeatures::duplicate() const
{
	return std::make_shared<SparsePolyFeatures>(*this);
}

void SparsePolyFeatures::init()
{
	SG_ADD(
	    &m_feat, "features", "Features in original space.");
	SG_ADD(
	    &m_degree, "degree", "Degree of the polynomial kernel.", ParameterProperties::HYPER);
	SG_ADD(&m_normalize, "normalize", "Normalize");
	SG_ADD(
	    &m_input_dimensions, "input_dimensions",
	    "Dimensions of the input space.");
	SG_ADD(
	    &m_output_dimensions, "output_dimensions",
	    "Dimensions of the feature space of the polynomial kernel.");

	m_normalization_values_len = get_num_vectors();
	/*m_parameters->add_vector(&m_normalization_values, &m_normalization_values_len,
			"m_normalization_values", "Norm of each training example");*/
	watch_param(
	    "m_normalization_values", &m_normalization_values,
	    &m_normalization_values_len);

	SG_ADD(&mask, "mask", "Mask.");
	SG_ADD(
	    &m_hash_bits, "m_hash_bits", "Number of bits in hash");
}
