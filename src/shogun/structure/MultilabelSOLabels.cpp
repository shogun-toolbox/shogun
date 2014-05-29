/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Written (W) 2014 Abinash Panda
 * Copyright (C) 2014 Abinash Panda
 */

#include <shogun/structure/MultilabelSOLabels.h>

using namespace shogun;

CMultilabelSOLabels::CMultilabelSOLabels() : CStructuredLabels()
{
	init();
	m_multilabel_labels = NULL;
}

CMultilabelSOLabels::CMultilabelSOLabels(int32_t num_classes)
	: CStructuredLabels()
{
	init();
	m_multilabel_labels = new CMultilabelLabels(num_classes);
}

CMultilabelSOLabels::CMultilabelSOLabels(int32_t num_labels, int32_t num_classes)
	: CStructuredLabels()
{
	init();
	m_multilabel_labels = new CMultilabelLabels(num_labels, num_classes);
}

CMultilabelSOLabels::CMultilabelSOLabels(CMultilabelLabels * multilabel_labels)
	: CStructuredLabels()
{
	init();
	m_multilabel_labels = multilabel_labels;
}

void CMultilabelSOLabels::init()
{
	SG_ADD((CSGObject **)&m_multilabel_labels, "multilabel_labels", "multilabel labels object",
	       MS_NOT_AVAILABLE);
	SG_ADD(&m_last_set_label, "last_set_label", "index of the last label added using add_label() method",
	       MS_NOT_AVAILABLE);

	m_last_set_label = 0;
}

CMultilabelSOLabels::~CMultilabelSOLabels()
{
	SG_UNREF(m_multilabel_labels);
}

void CMultilabelSOLabels::set_sparse_label(int32_t j, SGVector<int32_t> label)
{
	m_multilabel_labels->set_label(j, label);
}

void CMultilabelSOLabels::set_sparse_labels(SGVector<int32_t> * labels)
{
	m_multilabel_labels->set_labels(labels);
}

int32_t CMultilabelSOLabels::get_num_labels() const
{
	if (m_multilabel_labels == NULL)
	{
		return 0;
	}
	else
	{
		return m_multilabel_labels->get_num_labels();
	}
}

int32_t CMultilabelSOLabels::get_num_classes() const
{
	if (m_multilabel_labels == NULL)
	{
		return 0;
	}
	else
	{
		return m_multilabel_labels->get_num_classes();
	}
}

bool CMultilabelSOLabels::set_label(int32_t j, CStructuredData * label)
{
	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(label);
	m_multilabel_labels->set_label(j, slabel->get_data());
	return true;
}

CStructuredData * CMultilabelSOLabels::get_label(int32_t j)
{
	CSparseMultilabel * slabel = new CSparseMultilabel(m_multilabel_labels->get_label(j));
	SG_REF(slabel);
	return (CStructuredData *)slabel;
}

SGVector<int32_t> CMultilabelSOLabels::get_sparse_label(int32_t j)
{
	return m_multilabel_labels->get_label(j);
}

void CMultilabelSOLabels::ensure_valid(const char * context)
{
	m_multilabel_labels->ensure_valid(context);
}

SGVector<float64_t> CMultilabelSOLabels::to_dense(CStructuredData * label,
                int32_t dense_dim, float64_t d_true, float64_t d_false)
{
	CSparseMultilabel * slabel = CSparseMultilabel::obtain_from_generic(label);
	SGVector<int32_t> slabel_data = slabel->get_data();
	return CMultilabelLabels::to_dense<int32_t, float64_t>(&slabel_data,
	                dense_dim, d_true, d_false);
}

void CMultilabelSOLabels::add_label(CStructuredData * label)
{
	REQUIRE(m_last_set_label >= 0 && m_last_set_label < get_num_labels(),
	        "Only %d number of labels can be added.\n", get_num_labels());

	set_label(m_last_set_label, label);
	m_last_set_label++;
}

