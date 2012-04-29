/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2012 Chiyuan Zhang
 * Copyright (C) 2012 Chiyuan Zhang
 */

#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;


CMulticlassStrategy::CMulticlassStrategy()
	:m_train_labels(NULL), m_orig_labels(NULL), m_train_iter(0)
{
    SG_ADD(&m_num_classes, "num_classes", "Number of classes", MS_NOT_AVAILABLE);
}

void CMulticlassStrategy::train_start(CLabels *orig_labels, CLabels *train_labels)
{
	if (m_train_labels != NULL)
		SG_ERROR("Stop the previous training task before starting a new one!");
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
