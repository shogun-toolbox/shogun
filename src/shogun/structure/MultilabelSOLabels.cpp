/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Written (W) 2013 Thoralf Klein
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2013 Thoralf Klein and Zuse-Institute-Berlin (ZIB)
 * Copyright (C) 2014 Abinash Panda
 */

#include <shogun/structure/MultilabelSOLabels.h>

using namespace shogun;

MultilabelSOLabels::MultilabelSOLabels() : StructuredLabels()
{
	init();
	m_multilabel_labels = NULL;
}

MultilabelSOLabels::MultilabelSOLabels(int32_t num_classes)
	: StructuredLabels()
{
	init();
	m_multilabel_labels = std::make_shared<MultilabelLabels>(num_classes);
}

MultilabelSOLabels::MultilabelSOLabels(int32_t num_labels, int32_t num_classes)
	: StructuredLabels()
{
	init();
	m_multilabel_labels = std::make_shared<MultilabelLabels>(num_labels, num_classes);
}

MultilabelSOLabels::MultilabelSOLabels(std::shared_ptr<MultilabelLabels > multilabel_labels)
	: StructuredLabels()
{
	init();

	m_multilabel_labels = multilabel_labels;
}

void MultilabelSOLabels::init()
{
	SG_ADD((std::shared_ptr<SGObject>*)&m_multilabel_labels, "multilabel_labels", "multilabel labels object");
	SG_ADD(&m_last_set_label, "last_set_label", "index of the last label added using add_label() method");

	m_last_set_label = 0;
}

MultilabelSOLabels::~MultilabelSOLabels()
{

}

void MultilabelSOLabels::set_sparse_label(int32_t j, SGVector<int32_t> label)
{
	if (m_sdt == SDT_UNKNOWN)
	{
		m_sdt = SDT_SPARSE_MULTILABEL;
	}

	m_multilabel_labels->set_label(j, label);
}

void MultilabelSOLabels::set_sparse_labels(SGVector<int32_t> * labels)
{
	if (m_sdt == SDT_UNKNOWN)
	{
		m_sdt = SDT_SPARSE_MULTILABEL;
	}

	m_multilabel_labels->set_labels(labels);
}

int32_t MultilabelSOLabels::get_num_labels() const
{
	if (m_multilabel_labels == NULL)
	{
		return 0;
	}

	return m_multilabel_labels->get_num_labels();
}

int32_t MultilabelSOLabels::get_num_classes() const
{
	if (m_multilabel_labels == NULL)
	{
		return 0;
	}

	return m_multilabel_labels->get_num_classes();
}

std::shared_ptr<MultilabelLabels > MultilabelSOLabels::get_multilabel_labels()
{
	return m_multilabel_labels;
}

bool MultilabelSOLabels::set_label(int32_t j, std::shared_ptr<StructuredData > label)
{
	if (m_sdt == SDT_UNKNOWN)
	{
		m_sdt = label->get_structured_data_type();
	}

	auto slabel = label->as<SparseMultilabel>();
	m_multilabel_labels->set_label(j, slabel->get_data());
	return true;
}

std::shared_ptr<StructuredData > MultilabelSOLabels::get_label(int32_t j)
{
	auto slabel = std::make_shared<SparseMultilabel>(m_multilabel_labels->get_label(j));

	return slabel->as<StructuredData>();
}

SGVector<int32_t> MultilabelSOLabels::get_sparse_label(int32_t j)
{
	return m_multilabel_labels->get_label(j);
}

void MultilabelSOLabels::ensure_valid(const char * context)
{
	m_multilabel_labels->ensure_valid(context);
}

SGVector<float64_t> MultilabelSOLabels::to_dense(std::shared_ptr<StructuredData > label,
                int32_t dense_dim, float64_t d_true, float64_t d_false)
{
	auto slabel = label->as<SparseMultilabel>();
	SGVector<int32_t> slabel_data = slabel->get_data();
	return MultilabelLabels::to_dense<int32_t, float64_t>(&slabel_data,
	                dense_dim, d_true, d_false);
}

void MultilabelSOLabels::add_label(std::shared_ptr<StructuredData > label)
{
	require(m_last_set_label >= 0 && m_last_set_label < get_num_labels(),
	        "Only {} number of labels can be added.", get_num_labels());

	set_label(m_last_set_label, label);
	m_last_set_label++;
}
