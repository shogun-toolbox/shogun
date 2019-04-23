/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Chiyuan Zhang, Heiko Strathmann, Soeren Sonnenburg, Shell Hu,
 *          Sergey Lisitsyn
 */

#include <shogun/multiclass/MulticlassStrategy.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;


MulticlassStrategy::MulticlassStrategy()
	: SGObject()
{
	init();
}

MulticlassStrategy::MulticlassStrategy(EProbHeuristicType prob_heuris)
	: SGObject()
{
	init();

	m_prob_heuris=prob_heuris;
}

void MulticlassStrategy::init()
{
	m_rejection_strategy=NULL;
	m_train_labels=NULL;
	m_orig_labels=NULL;
	m_train_iter=0;
	m_prob_heuris=PROB_HEURIS_NONE;
	m_num_classes=0;

	SG_ADD(
	    (std::shared_ptr<SGObject>*)&m_rejection_strategy, "rejection_strategy",
	    "Strategy of rejection");
	SG_ADD(&m_num_classes, "num_classes", "Number of classes");
	SG_ADD_OPTIONS(
	    (machine_int_t*)&m_prob_heuris, "prob_heuris",
	    "Probability estimation heuristics", ParameterProperties::NONE,
	    SG_OPTIONS(PROB_HEURIS_NONE, OVA_NORM, OVA_SOFTMAX, OVO_PRICE, OVO_HASTIE,
	    OVO_HAMAMURA));
}

void MulticlassStrategy::train_start(std::shared_ptr<MulticlassLabels >orig_labels, std::shared_ptr<BinaryLabels >train_labels)
{
	if (m_train_labels != NULL)
		error("Stop the previous training task before starting a new one!");
	m_train_labels=train_labels;

	m_orig_labels=orig_labels;
	m_train_iter=0;
}

SGVector<int32_t> MulticlassStrategy::train_prepare_next()
{
	m_train_iter++;
	return SGVector<int32_t>();
}

void MulticlassStrategy::train_stop()
{


    m_train_labels = NULL;
    m_orig_labels = NULL;
}
