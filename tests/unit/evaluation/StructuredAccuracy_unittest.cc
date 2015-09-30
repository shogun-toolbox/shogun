/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Copyright(C) 2014 Abinash Panda
 * Written(W) 2014 Abinash Panda
 */

#include <shogun/evaluation/StructuredAccuracy.h>
#include <shogun/labels/MultilabelLabels.h>
#include <shogun/lib/SGVector.h>
#include <shogun/structure/MultilabelSOLabels.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(StructuredAccuracy, evaluate_multilabel_so_labels)
{
	SGVector<int32_t> lab_1(4);
	SGVector<int32_t> lab_2(4);

	for (index_t i = 0; i < lab_1.vlen; i++)
	{
		lab_1[i] = i;
		lab_2[i] = i;
	}

	SGVector<int32_t> lab_3(5);

	for (index_t i = 0; i < lab_3.vlen; i++)
	{
		lab_3[i] = i;
	}

	CMultilabelSOLabels * m_labels_1 = new CMultilabelSOLabels(1, 5);
	SG_REF(m_labels_1);
	m_labels_1->set_sparse_label(0, lab_1);

	CMultilabelSOLabels * m_labels_2 = new CMultilabelSOLabels(1, 5);
	SG_REF(m_labels_2);
	m_labels_2->set_sparse_label(0, lab_2);

	CMultilabelSOLabels * m_labels_3 = new CMultilabelSOLabels(1, 5);
	SG_REF(m_labels_3);
	m_labels_3->set_sparse_label(0, lab_3);

	CStructuredAccuracy * evaluator = new CStructuredAccuracy();
	SG_REF(evaluator);

	float64_t acc;

	acc = evaluator->evaluate(m_labels_2, m_labels_1);
	EXPECT_NEAR(acc, 1.0, 1E-7);

	acc = evaluator->evaluate(m_labels_3, m_labels_1);
	EXPECT_NEAR(acc, 0.8, 1E-7);

	SG_UNREF(m_labels_1);
	SG_UNREF(m_labels_2);
	SG_UNREF(m_labels_3);
	SG_UNREF(evaluator);
}

