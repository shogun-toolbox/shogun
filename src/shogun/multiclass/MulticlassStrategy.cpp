/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Heiko Strathmann, Soeren Sonnenburg, Shell Hu, 
 *          Sergey Lisitsyn
 */

#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/mathematics/Math.h>

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
	SG_ADD((machine_int_t*)&m_prob_heuris, "prob_heuris", "Probability estimation heuristics", MS_NOT_AVAILABLE);
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
