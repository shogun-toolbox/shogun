/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Heiko Strathmann,
 *          Evangelos Anagnostopoulos
 */

#include <shogun/evaluation/PRCEvaluation.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/RegressionLabels.h>
#include <shogun/mathematics/Math.h>

using namespace shogun;

PRCEvaluation::PRCEvaluation() : BinaryClassEvaluation(), m_computed(false)
{
	m_PRC_graph = SGMatrix<float64_t>();
	m_thresholds = SGVector<float64_t>();
	m_auPRC = 0.0;
	watch_method("PRC", &PRCEvaluation::get_PRC);
	watch_method("thresholds", &PRCEvaluation::get_thresholds);
	watch_method("auPRC", &PRCEvaluation::get_auPRC);
};

PRCEvaluation::~PRCEvaluation()
{
}

float64_t PRCEvaluation::evaluate(std::shared_ptr<Labels> predicted, std::shared_ptr<Labels> ground_truth)
{
	ASSERT(predicted && ground_truth)
	ASSERT(predicted->get_num_labels() == ground_truth->get_num_labels())
	ASSERT(predicted->get_label_type() == LT_BINARY)
	ASSERT(ground_truth->get_label_type() == LT_BINARY)
	ground_truth->ensure_valid();

	// number of true positive examples
	float64_t tp = 0.0;
	int32_t i;

	// total number of positive labels in predicted
	int32_t pos_count = 0;

	// initialize number of labels and labels
	SGVector<float64_t> orig_labels = predicted->get_values();
	int32_t length = orig_labels.vlen;
	float64_t* labels =
	    SGVector<float64_t>::clone_vector(orig_labels.vector, length);

	// get indexes for sort
	int32_t* idxs = SG_MALLOC(int32_t, length);
	for (i = 0; i < length; i++)
		idxs[i] = i;

	// sort indexes by labels ascending
	Math::qsort_backward_index(labels, idxs, length);

	// clean and initialize graph and auPRC
	SG_FREE(labels);
	m_PRC_graph = SGMatrix<float64_t>(2, length);
	m_thresholds = SGVector<float64_t>(length);
	m_auPRC = 0.0;

	// get total numbers of positive and negative labels
	for (i = 0; i < length; i++)
	{
		if (ground_truth->get_value(i) > 0)
			pos_count++;
	}

	// assure number of positive examples is >0
	ASSERT(pos_count > 0)

	// create PRC curve
	for (i = 0; i < length; i++)
	{
		// update number of true positive examples
		if (ground_truth->get_value(idxs[i]) > 0)
			tp += 1.0;

		// precision (x)
		m_PRC_graph[2 * i] = tp / float64_t(i + 1);
		// recall (y)
		m_PRC_graph[2 * i + 1] = tp / float64_t(pos_count);

		m_thresholds[i] = predicted->get_value(idxs[i]);
	}

	// calc auRPC using area under curve
	m_auPRC = Math::area_under_curve(m_PRC_graph.matrix, length, true);

	// set computed indicator
	m_computed = true;

	SG_FREE(idxs);
	return m_auPRC;
}

SGMatrix<float64_t> PRCEvaluation::get_PRC() const
{
	if (!m_computed)
		error("Uninitialized, please call evaluate first");

	return m_PRC_graph;
}

SGVector<float64_t> PRCEvaluation::get_thresholds() const
{
	if (!m_computed)
		error("Uninitialized, please call evaluate first");

	return m_thresholds;
}

float64_t PRCEvaluation::get_auPRC() const
{
	if (!m_computed)
		error("Uninitialized, please call evaluate first");

	return m_auPRC;
}
