#include <shogun/features/IndexFeatures.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

CIndexFeatures::CIndexFeatures()
{
	init();
	num_vectors = 0;
}

CIndexFeatures::CIndexFeatures(SGVector<index_t> vector) : CDummyFeatures(vector.vlen)
{
	init();
}

CIndexFeatures::CIndexFeatures(const CIndexFeatures &orig) : CDummyFeatures(orig.feature_index.vlen)
{
	init();
	set_feature_index(orig.feature_index);
	if (orig.m_subset_stack != NULL)
	{
		SG_UNREF(m_subset_stack);
		m_subset_stack=new CSubsetStack(*orig.m_subset_stack);
		SG_REF(m_subset_stack);
	}
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
	return new CIndexFeatures(*this);
}

inline EFeatureType CIndexFeatures::get_feature_type() const
{
	return F_INT;
}

EFeatureClass CIndexFeatures::get_feature_class() const
{
	return C_INDEX;
}

SGVector<index_t> CIndexFeatures::get_feature_index()
{
	if (!m_subset_stack->has_subsets())
		return feature_index;

	SGVector<index_t> subvector(get_num_vectors());

	/* copy a subset vector wise */
	for (int32_t i=0; i<subvector.vlen; ++i)
	{
		int32_t real_i = m_subset_stack->subset_idx_conversion(i);
		subvector[i] = feature_index[real_i];
	}

	return subvector;
}

void CIndexFeatures::set_feature_index(SGVector<index_t> vector)
{
	m_subset_stack->remove_all_subsets();
	free_feature_index();
	feature_index = vector;
	num_vectors = vector.vlen;
}

void CIndexFeatures::free_feature_index()
{
	m_subset_stack->remove_all_subsets();
	feature_index=SGVector<index_t>();
	num_vectors = 0;
}

void CIndexFeatures::init()
{
	SG_ADD(&num_vectors, "num_vectors", "Number of vectors.", MS_NOT_AVAILABLE);
	SG_ADD(&feature_index, "feature_index",
				"Vector of feature index.", MS_NOT_AVAILABLE);
}
