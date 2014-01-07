/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Written (W) 2013 Shell Hu and Heiko Strathmann
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <multiclass/MulticlassStrategy.h>
#include <mathematics/Math.h>

using namespace shogun;


CMulticlassStrategy::CMulticlassStrategy()
	: CSGObject()
{
	init();
}

CMulticlassStrategy::CMulticlassStrategy(EProbHeuristicType prob_heuris)
	: CSGObject()
{
	init();

	m_prob_heuris=prob_heuris;
}

void CMulticlassStrategy::init()
{
	m_rejection_strategy=NULL;
	m_train_labels=NULL;
	m_orig_labels=NULL;
	m_train_iter=0;
	m_prob_heuris=PROB_HEURIS_NONE;
	m_num_classes=0;

	SG_ADD((CSGObject**)&m_rejection_strategy, "rejection_strategy", "Strategy of rejection", MS_NOT_AVAILABLE);
	SG_ADD(&m_num_classes, "num_classes", "Number of classes", MS_NOT_AVAILABLE);
	//SG_ADD((machine_int_t*)&m_prob_heuris, "prob_heuris", "Probability estimation heuristics", MS_NOT_AVAILABLE);

	SG_WARNING("%s::CMulticlassStrategy(): register parameters!\n", get_name());
}

void CMulticlassStrategy::train_start(CMulticlassLabels *orig_labels, CBinaryLabels *train_labels)
{
	if (m_train_labels != NULL)
		SG_ERROR("Stop the previous training task before starting a new one!")
	SG_REF(train_labels);
	m_train_labels=train_labels;
	SG_REF(orig_labels);
	m_orig_labels=orig_labels;
	m_train_iter=0;
}

SGVector<int32_t> CMulticlassStrategy::train_prepare_next()
{
	m_train_iter++;
	return SGVector<int32_t>();
}

void CMulticlassStrategy::train_stop()
{
	SG_UNREF(m_train_labels);
	SG_UNREF(m_orig_labels);
    m_train_labels = NULL;
    m_orig_labels = NULL;
}
