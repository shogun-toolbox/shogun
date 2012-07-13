/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/tree/RelaxedTreeUtil.h>
#include <shogun/multiclass/tree/RelaxedTree.h>


using namespace shogun;

CRelaxedTree::CRelaxedTree()
	:m_feats(NULL), m_machine_for_confusion_matrix(NULL), m_num_classes(0)
{
}

CRelaxedTree::~CRelaxedTree()
{
	SG_UNREF(m_feats);
	SG_UNREF(m_machine_for_confusion_matrix);
}

CMulticlassLabels* CRelaxedTree::apply_multiclass(CFeatures* data)
{
	return NULL;
}

bool CRelaxedTree::train_machine(CFeatures* data)
{
	if (m_machine_for_confusion_matrix == NULL)
		SG_ERROR("Call set_machine_for_confusion_matrix before training");

	if (data)
	{
		CDenseFeatures<float64_t> *feats = dynamic_cast<CDenseFeatures<float64_t>*>(data);
		if (feats == NULL)
			SG_ERROR("Require non-NULL dense features of float64_t\n");
		set_features(feats);
	}

	CMulticlassLabels *lab = dynamic_cast<CMulticlassLabels *>(m_labels);

	RelaxedTreeUtil util;
	SGMatrix<float64_t> conf_mat = util.estimate_confusion_matrix(m_machine_for_confusion_matrix,
			m_feats, lab, m_num_classes);

	return false;
}
