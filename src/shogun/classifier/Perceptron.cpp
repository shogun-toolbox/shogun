/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Giovanni De Toni, 
 *          Michele Mazzoni, Heiko Strathmann, Fernando Iglesias
 */

#include <shogun/base/range.h>
#include <shogun/classifier/Perceptron.h>
#include <shogun/features/iterators/DotIterator.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/labels/Labels.h>
#include <shogun/lib/Signal.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>

using namespace shogun;

CPerceptron::CPerceptron()
: CLinearMachine()
{
	init();
}

CPerceptron::CPerceptron(CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	init();
	set_features(traindat);
	set_labels(trainlab);
}

void CPerceptron::init()
{
	max_iter = 1000;
	learn_rate = 0.1;
	m_initialize_hyperplane = true;
	SG_ADD(&max_iter, "initialize_hyperplane", "Whether to initialize hyperplane.", MS_AVAILABLE);
	SG_ADD(&max_iter, "max_iter", "Maximum number of iterations.", MS_AVAILABLE);
	SG_ADD(&learn_rate, "learn_rate", "Learning rate.", MS_AVAILABLE);
}

CPerceptron::~CPerceptron()
{
}

bool CPerceptron::train_machine(CFeatures* data)
{
	ASSERT(m_labels)

	if (data)
	{
		if (!data->has_property(FP_DOT))
			SG_ERROR("Specified features are not of type CDotFeatures\n")
		set_features((CDotFeatures*) data);
	}

	ASSERT(features)
	bool converged=false;
	int32_t iter=0;
	SGVector<int32_t> train_labels = binary_labels(m_labels)->get_int_labels();
	int32_t num_feat=features->get_dim_feature_space();
	int32_t num_vec=features->get_num_vectors();

	ASSERT(num_vec==train_labels.vlen)
	SGVector<float64_t> output(num_vec);

	SGVector<float64_t> w = get_w();
	if (m_initialize_hyperplane)
	{
		w = SGVector<float64_t>(num_feat);
		//start with uniform w, bias=0
		bias=0;
		linalg::add_scalar(w, 1.0 / num_feat);
	}

	//loop till we either get everything classified right or reach max_iter
	while (!(cancel_computation()) && (!converged && iter < max_iter))
	{
		converged=true;
		auto iter_train_labels = train_labels.begin();
		auto iter_output = output.begin();
		for (const auto& v : DotIterator(features))
		{
			const auto true_label = *(iter_train_labels++);
			auto& predicted_label = *(iter_output++);

			predicted_label = v.dot(w) + bias;

			if (CMath::sign<float64_t>(predicted_label) != true_label)
			{
				converged = false;
				const auto gradient = learn_rate * true_label;
				bias += gradient;
				v.add(gradient, w);
			}
		}

		iter++;
	}

	if (converged)
		SG_INFO("Perceptron algorithm converged after %d iterations.\n", iter)
	else
		SG_WARNING("Perceptron algorithm did not converge after %d iterations.\n", max_iter)

	set_w(w);

	return converged;
}

void CPerceptron::set_initialize_hyperplane(bool initialize_hyperplane)
{
	m_initialize_hyperplane = initialize_hyperplane;
}

bool CPerceptron::get_initialize_hyperplane()
{
	return m_initialize_hyperplane;
}
