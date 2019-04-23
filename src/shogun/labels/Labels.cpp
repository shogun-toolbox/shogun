/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Weijie Lin, Thoralf Klein,
 *          Leon Kuchenbecker
 */

#include <shogun/io/SGIO.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/lib/common.h>

using namespace shogun;

Labels::Labels()
	: SGObject()
{
	init();
}

Labels::Labels(const Labels& orig)
    : SGObject(orig), m_current_values(orig.m_current_values)
{
	init();

	if (orig.m_subset_stack != NULL)
	{

		m_subset_stack = std::make_shared<SubsetStack>(*orig.m_subset_stack);

	}
}

Labels::~Labels()
{

}

void Labels::init()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_subset_stack, "subset_stack",
	       "Current subset stack");
	SG_ADD(
	    &m_current_values, "current_values", "current active value vector");
	m_subset_stack = std::make_shared<SubsetStack>();

}

void Labels::add_subset(SGVector<index_t> subset)
{
	m_subset_stack->add_subset(subset);
}

void Labels::add_subset_in_place(SGVector<index_t> subset)
{
	m_subset_stack->add_subset_in_place(subset);
}

void Labels::remove_subset()
{
	m_subset_stack->remove_subset();
}

void Labels::remove_all_subsets()
{
	m_subset_stack->remove_all_subsets();
}

std::shared_ptr<SubsetStack> Labels::get_subset_stack()
{

	return m_subset_stack;
}

float64_t Labels::get_value(int32_t idx)
{
	ASSERT(m_current_values.vector && idx < get_num_labels())
	int32_t real_num = m_subset_stack->subset_idx_conversion(idx);
	return m_current_values.vector[real_num];
}

void Labels::set_value(float64_t value, int32_t idx)
{

	REQUIRE(m_current_values.vector, "%s::set_value(%f, %d): No values vector"
	        " set!\n", get_name(), value, idx);
	REQUIRE(get_num_labels(), "%s::set_value(%f, %d): Number of values is "
	        "zero!\n", get_name(), value, idx);

	int32_t real_num = m_subset_stack->subset_idx_conversion(idx);
	m_current_values.vector[real_num] = value;
}

void Labels::set_values(SGVector<float64_t> values)
{
	if (m_current_values.vlen != 0 && m_current_values.vlen != get_num_labels())
	{
		SG_ERROR("length of value values should match number of labels or"
		         " have zero length (len(labels)=%d, len(values)=%d)\n",
		         get_num_labels(), values.vlen);
	}

	m_current_values = values;
}

SGVector<float64_t> Labels::get_values() const
{
	return m_current_values;
}
