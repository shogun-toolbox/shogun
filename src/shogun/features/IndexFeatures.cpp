#include <shogun/features/IndexFeatures.h>
#include <shogun/base/Parameter.h>

using namespace shogun;

IndexFeatures::IndexFeatures()
{
	init();
}

IndexFeatures::IndexFeatures(SGVector<index_t> feature_index)
{
	init();
	set_feature_index(feature_index);
}

IndexFeatures::~IndexFeatures()
{
}

int32_t IndexFeatures::get_num_vectors() const
{
	return m_subset_stack->has_subsets() ? m_subset_stack->get_size() : num_vectors;
}

std::shared_ptr<Features> IndexFeatures::duplicate() const
{
	auto indexfeature_dup =
			std::make_shared<IndexFeatures>(m_feature_index.clone());
	if (m_subset_stack != NULL)
	{

		indexfeature_dup->m_subset_stack=std::make_shared<SubsetStack>(*m_subset_stack);

	}
	return indexfeature_dup;
}

EFeatureType IndexFeatures::get_feature_type() const
{
	return F_ANY;
}

EFeatureClass IndexFeatures::get_feature_class() const
{
	return C_INDEX;
}

SGVector<index_t> IndexFeatures::get_feature_index()
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

void IndexFeatures::set_feature_index(SGVector<index_t> feature_index)
{
	m_subset_stack->remove_all_subsets();
	free_feature_index();
	m_feature_index = feature_index;
	num_vectors = feature_index.vlen;
}

void IndexFeatures::free_feature_index()
{
	m_subset_stack->remove_all_subsets();
	m_feature_index=SGVector<index_t>();
	num_vectors = 0;
}

void IndexFeatures::init()
{
	num_vectors = 0;
	SG_ADD(&m_feature_index, "m_feature_index",
				"Vector of feature index.");
}
