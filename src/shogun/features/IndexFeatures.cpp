#include <shogun/features/IndexFeatures.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CIndexFeatures::CIndexFeatures()
{
	init();
}

CIndexFeatures::CIndexFeatures(SGVector<index_t> feature_index)
{
	init();
	set_feature_index(feature_index);
}

CIndexFeatures::~CIndexFeatures()
{
}

int32_t CIndexFeatures::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : num_vectors;
}

CFeatures* CIndexFeatures::duplicate() const
{
	SG_NOTIMPLEMENTED
	return NULL;
}

EFeatureType CIndexFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass CIndexFeatures::get_feature_class() const
{
	return C_INDEX;
}

SGVector<index_t> CIndexFeatures::get_feature_index()
{
	if (!m_subset_stack->has_subsets())
		return m_feature_index;

	SGVector<index_t> sub_feature_index(get_num_vectors());

	/* copy a subset vector wise */
	for (int32_t i=0; i<sub_feature_index.vlen; ++i)
	{
		int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		sub_feature_index[i] = m_feature_index[real_i];
	}

	return sub_feature_index;
}

void CIndexFeatures::set_feature_index(SGVector<index_t> feature_index)
{
	m_subset_stack->remove_all_subsets();
	free_feature_index();
	m_feature_index = feature_index;
	num_vectors = feature_index.vlen;
}

void CIndexFeatures::free_feature_index()
{
	m_subset_stack->remove_all_subsets();
	m_feature_index=SGVector<index_t>();
	num_vectors = 0;
}

void CIndexFeatures::init()
{
	num_vectors = 0;
	SG_ADD(&m_feature_index, "m_feature_index",
				"Vector of feature index.", MS_NOT_AVAILABLE);
}
