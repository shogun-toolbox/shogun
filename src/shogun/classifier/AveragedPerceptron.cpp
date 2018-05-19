/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Soeren Sonnenburg, Sergey Lisitsyn, Giovanni De Toni,
 *          Michele Mazzoni
 */

#include <shogun/classifier/AveragedPerceptron.h>
#include <shogun/labels/Labels.h>
#include <shogun/mathematics/Math.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/base/progress.h>
#include <shogun/lib/Signal.h>

using namespace shogun;

CAveragedPerceptron::CAveragedPerceptron()
: CLinearMachine()
{
	init();

}

CAveragedPerceptron::CAveragedPerceptron(CDotFeatures* traindat, CLabels* trainlab)
: CLinearMachine()
{
	set_features(traindat);
	set_labels(trainlab);
	init();
}

CAveragedPerceptron::~CAveragedPerceptron()
{
}

void CAveragedPerceptron::init()
{
	max_iter = 1000;
	learn_rate = 0.1;
	SG_ADD(&max_iter, "max_iter", "Maximum number of iterations.", MS_AVAILABLE);
	SG_ADD(&learn_rate, "learn_rate", "Learning rate.", MS_AVAILABLE);
}

bool CAveragedPerceptron::train_machine(CFeatures* data)
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
	SGVector<float64_t> w(num_feat);
	float64_t* tmp_w=SG_MALLOC(float64_t, num_feat);
	memset(tmp_w, 0, sizeof(float64_t)*num_feat);
	float64_t* output=SG_MALLOC(float64_t, num_vec);

	//start with uniform w, bias=0, tmp_bias=0
	bias=0;
	float64_t tmp_bias=0;
	for (int32_t i=0; i<num_feat; i++)
		w[i]=1.0/num_feat;


  auto pb = progress(range(max_iter));
	//loop till we either get everything classified right or reach max_iter
	while ((!converged && iter < max_iter))
	{
		COMPUTATION_CONTROLLERS
		converged=true;
		SG_INFO("Iteration Number : %d of max %d\n", iter, max_iter);

		for (int32_t i=0; i<num_vec; i++)
		{
			output[i] = features->dense_dot(i, w.vector, w.vlen) + bias;

			if (CMath::sign<float64_t>(output[i]) != train_labels.vector[i])
			{
				converged=false;
				bias+=learn_rate*train_labels.vector[i];
				features->add_to_dense_vec(learn_rate*train_labels.vector[i], i, w.vector, w.vlen);
			}

			// Add current w to tmp_w, and current bias to tmp_bias
			// To calculate the sum of each iteration's w, bias
			for (int32_t j=0; j<num_feat; j++)
				tmp_w[j]+=w[j];
			tmp_bias+=bias;
		}
		iter++;
		pb.print_progress();
	}
	pb.complete();
	if (converged)
		SG_INFO("Averaged Perceptron algorithm converged after %d iterations.\n", iter)
	else
		SG_WARNING("Averaged Perceptron algorithm did not converge after %d iterations.\n", max_iter)

	// calculate and set the average paramter of w, bias
	for (int32_t i=0; i<num_feat; i++)
		w[i]=tmp_w[i]/(num_vec*iter);
	bias=tmp_bias/(num_vec*iter);

	SG_FREE(output);
	SG_FREE(tmp_w);

	set_w(w);

	return converged;
}
