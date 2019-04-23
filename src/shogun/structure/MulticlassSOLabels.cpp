/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Thoralf Klein, Sanuj Sharma, Fernando Iglesias, Soeren Sonnenburg, 
 *          Bjoern Esser
 */

#include <shogun/structure/MulticlassSOLabels.h>

using namespace shogun;

MulticlassSOLabels::MulticlassSOLabels()
: StructuredLabels(), m_labels_vector(16)
{
	init();
}

MulticlassSOLabels::MulticlassSOLabels(int32_t num_labels)
: StructuredLabels(), m_labels_vector(num_labels)
{
	init();
}

MulticlassSOLabels::MulticlassSOLabels(SGVector< float64_t > const src)
: StructuredLabels(src.vlen), m_labels_vector(src.vlen)
{
	init();

	m_num_classes = Math::max(src.vector, src.vlen) + 1;
	m_labels_vector.resize_vector(src.vlen);

	for ( int32_t i = 0 ; i < src.vlen ; ++i )
	{
		if ( src[i] < 0 || src[i] >= m_num_classes )
			SG_ERROR("Found label out of {0, 1, 2, ..., num_classes-1}")
		else
			add_label( std::make_shared<RealNumber>(src[i]) );
	}

	//TODO check that every class has at least one example
}

MulticlassSOLabels::~MulticlassSOLabels()
{
}

std::shared_ptr<StructuredData> MulticlassSOLabels::get_label(int32_t idx)
{
	// ensure_valid("CMulticlassSOLabels::get_label(int32_t)");
	if ( idx < 0 || idx >= get_num_labels() )
		SG_ERROR("Index must be inside [0, num_labels-1]\n")

	return std::static_pointer_cast<StructuredData>( std::make_shared<RealNumber>(m_labels_vector[idx]));
}

void MulticlassSOLabels::add_label(std::shared_ptr<StructuredData> label)
{
        
        float64_t value = label->as<RealNumber>()->value;
        

	//ensure_valid_sdt(label);
	if (m_num_labels_set >= m_labels_vector.vlen)
	{
		m_labels_vector.resize_vector(m_num_labels_set + 16);
	}


	m_labels_vector[m_num_labels_set] = value;
	m_num_labels_set++;
}

bool MulticlassSOLabels::set_label(int32_t idx, std::shared_ptr<StructuredData> label)
{
        
        float64_t value = label->as<RealNumber>()->value;
        

	// ensure_valid_sdt(label);
	int32_t real_idx = m_subset_stack->subset_idx_conversion(idx);

	if ( real_idx < get_num_labels() )
	{
		m_labels_vector[real_idx] = value;
		return true;
	}
	else
	{
		return false;
	}
}

int32_t MulticlassSOLabels::get_num_labels() const
{
	return m_num_labels_set;
}

void MulticlassSOLabels::init()
{
	SG_ADD(&m_num_classes, "m_num_classes", "The number of classes");
	SG_ADD(&m_num_labels_set, "m_num_labels_set", "The number of assigned labels");
	SG_ADD(
	    &m_labels_vector, "labels_vector", "The labels vector");

	m_num_classes = 0;
	m_num_labels_set = 0;
}
