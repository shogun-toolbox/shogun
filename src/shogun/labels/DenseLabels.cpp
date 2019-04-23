/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Heiko Strathmann, Sergey Lisitsyn,
 *          Fernando Iglesias, Michele Mazzoni, Chiyuan Zhang
 */

#include <shogun/base/Parameter.h>
#include <shogun/io/File.h>
#include <shogun/io/SGIO.h>
#include <shogun/labels/DenseLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/common.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

DenseLabels::DenseLabels()
: Labels()
{
	init();
}

DenseLabels::DenseLabels(int32_t num_lab)
: Labels()
{
	init();
	m_labels = SGVector<float64_t>(num_lab);
	m_current_values=SGVector<float64_t>(num_lab);
}

DenseLabels::DenseLabels(const DenseLabels& orig)
    : Labels(orig), m_labels(orig.m_labels)
{
	init();
}

DenseLabels::DenseLabels(std::shared_ptr<File> loader)
: Labels()
{
	init();
	load(loader);
}

DenseLabels::~DenseLabels()
{
}

void DenseLabels::init()
{
	SG_ADD(&m_labels, "labels", "The labels.");
}

void DenseLabels::set_to_one()
{
	set_to_const(1.0);
}

void DenseLabels::zero()
{
	set_to_const(0.0);
}

void DenseLabels::set_to_const(float64_t c)
{
	ASSERT(m_labels.vector)
	index_t subset_size=get_num_labels();
	for (int32_t i=0; i<subset_size; i++)
	{
		m_labels.vector[m_subset_stack->subset_idx_conversion(i)]=c;
		m_current_values.vector[m_subset_stack->subset_idx_conversion(i)]=c;
	}
}

void DenseLabels::set_labels(SGVector<float64_t> v)
{
	if (m_subset_stack->has_subsets())
		error("A subset is set, cannot set labels");

	m_labels = v;
}

SGVector<float64_t> DenseLabels::get_labels() const
{
	if (m_subset_stack->has_subsets())
		return get_labels_copy();

	return m_labels;
}

SGVector<float64_t> DenseLabels::get_labels_copy() const
{
	if (!m_subset_stack->has_subsets())
		return m_labels.clone();

	index_t num_labels = get_num_labels();
	SGVector<float64_t> result(num_labels);

	/* copy element wise because of possible subset */
	for (index_t i=0; i<num_labels; i++)
		result[i] = get_label(i);

	return result;
}

SGVector<int32_t> DenseLabels::get_int_labels() const
{
	SGVector<int32_t> intlab(get_num_labels());

	for (int32_t i=0; i<get_num_labels(); i++)
		intlab.vector[i] = get_int_label(i);

	return intlab;
}

void DenseLabels::set_int_labels(SGVector<int32_t> lab)
{
	if (m_subset_stack->has_subsets())
		error("set_int_labels() is not possible on subset");

	m_labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, lab.vector[i]);
}

#if !defined(SWIGJAVA) && !defined(SWIGCSHARP)
void DenseLabels::set_int_labels(SGVector<int64_t> lab)
{
	if (m_subset_stack->has_subsets())
		error("set_int_labels() is not possible on subset");

	m_labels = SGVector<float64_t>(lab.vlen);

	for (int32_t i=0; i<lab.vlen; i++)
		set_int_label(i, lab.vector[i]);
}
#endif

bool DenseLabels::is_valid() const
{
	return (m_labels.data() != nullptr) && (m_labels.size() > 0);
}

void DenseLabels::ensure_valid(const char* context)
{
	require(is_valid(), "Labels cannot be empty!");
}

void DenseLabels::load(std::shared_ptr<File> loader)
{
	remove_subset();
	m_labels = SGVector<float64_t>();
	m_labels.load(loader);
}

void DenseLabels::save(std::shared_ptr<File> writer)
{
	if (m_subset_stack->has_subsets())
		error("save() is not possible on subset");

	m_labels.save(writer);
}

bool DenseLabels::set_label(int32_t idx, float64_t label)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	if (m_labels.vector && real_num<get_num_labels())
	{
		m_labels.vector[real_num]=label;
		return true;
	}
	else
		return false;
}

bool DenseLabels::set_int_label(int32_t idx, int32_t label)
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	if (m_labels.vector && real_num<get_num_labels())
	{
		m_labels.vector[real_num] = (float64_t)label;
		return true;
	}
	else
		return false;
}

float64_t DenseLabels::get_label(int32_t idx) const
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_labels.vector && idx<get_num_labels())
	return m_labels.vector[real_num];
}

int32_t DenseLabels::get_int_label(int32_t idx) const
{
	int32_t real_num=m_subset_stack->subset_idx_conversion(idx);
	ASSERT(m_labels.vector && idx<get_num_labels())
	if (m_labels.vector[real_num] != float64_t((int32_t(m_labels.vector[real_num]))))
		error("label[{}]={:g} is not an integer", m_labels.vector[real_num]);

	return int32_t(m_labels.vector[real_num]);
}

int32_t DenseLabels::get_num_labels() const
{
	return m_subset_stack->has_subsets()
			? m_subset_stack->get_size() : m_labels.vlen;
}
