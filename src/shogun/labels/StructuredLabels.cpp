/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Fernando Iglesias, Thoralf Klein, Evgeniy Andreev, 
 *          Soeren Sonnenburg, Bjoern Esser
 */

#include <shogun/labels/StructuredLabels.h>

using namespace shogun;

StructuredLabels::StructuredLabels()
: Labels()
{
	init();
}

StructuredLabels::StructuredLabels(int32_t num_labels)
: Labels()
{
	init();
	m_labels.reserve(num_labels);
	
}

StructuredLabels::~StructuredLabels()
{
	
}

bool StructuredLabels::is_valid() const
{
	return true;
}

void StructuredLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Non-valid StructuredLabels in {}", context);
}

std::vector<std::shared_ptr<StructuredData>> StructuredLabels::get_labels() const
{
	
	return m_labels;
}

std::shared_ptr<StructuredData> StructuredLabels::get_label(int32_t idx)
{
	ensure_valid("StructuredLabels::get_label(int32_t)");
	if ( idx < 0 || idx >= get_num_labels() )
		error("Index must be inside [0, num_labels-1]");

	return m_labels[idx];
}

void StructuredLabels::add_label(std::shared_ptr<StructuredData> label)
{
	ensure_valid_sdt(label);
	m_labels.push_back(label);
}

bool StructuredLabels::set_label(int32_t idx, std::shared_ptr<StructuredData> label)
{
	ensure_valid_sdt(label);
	int32_t real_idx = m_subset_stack->subset_idx_conversion(idx);

	if ( real_idx < get_num_labels() )
	{
		m_labels[real_idx] = label;
		return true;
	}
	else
	{
		return false;
	}
}

int32_t StructuredLabels::get_num_labels() const
{
	return m_labels.size();
}

void StructuredLabels::init()
{
	SG_ADD(&m_labels, "m_labels", "The labels");

	m_labels.clear();
	m_sdt = SDT_UNKNOWN;
}

void StructuredLabels::ensure_valid_sdt(const std::shared_ptr<StructuredData>& label)
{
	if ( m_sdt == SDT_UNKNOWN )
	{
		m_sdt = label->get_structured_data_type();
	}
	else
	{
		require(label->get_structured_data_type() == m_sdt, "All the labels must "
				"belong to the same StructuredData child class");
	}
}
