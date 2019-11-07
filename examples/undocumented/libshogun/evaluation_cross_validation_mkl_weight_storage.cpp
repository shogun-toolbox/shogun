/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Giovanni De Toni, Heiko Strathmann, Thoralf Klein
 */

#include <shogun/base/range.h>
#include <shogun/classifier/mkl/MKLClassification.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/lib/parameter_observers/ParameterObserverCV.h>
#include <shogun/mathematics/Statistics.h>

using namespace shogun;

void gen_rand_data(SGVector<float64_t> lab, SGMatrix<float64_t> feat,
		float64_t dist)
{
	index_t dims=feat.num_rows;
	index_t num=lab.vlen;

	for (int32_t i=0; i<num; i++)
	{
		if (i<num/2)
		{
			lab[i]=-1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=Math::random(0.0, 1.0)+dist;
		}
		else
		{
			lab[i]=1.0;

			for (int32_t j=0; j<dims; j++)
				feat(j, i)=Math::random(0.0, 1.0)-dist;
		}
	}
	lab.display_vector("lab");
	feat.display_matrix("feat");
}

SGMatrix<float64_t> calculate_weights(
    ParameterObserverCV& obs, int32_t folds, int32_t run, int32_t len)
{
	int32_t column = 0;
	SGMatrix<float64_t> weights(len, folds * run);
	for (auto o : range(obs.get_num_observations()))
	{
		auto obs_storage = obs.get_observation(o);
		for (auto i : range(obs_storage->get<index_t>("num_folds")))
		{
			auto fold = obs_storage->get("folds", i);
			auto machine =
			    fold->get("trained_machine")->as<MKLClassification>();
			auto k = machine->get_kernel()->as<CombinedKernel>();
			auto w = k->get_subkernel_weights();

			/* Copy the weights inside the matrix */
			/* Each of the columns will represent a set of weights */
			for (auto j = 0; j < w.size(); j++)
			{
				weights.set_element(w[j], j, column);
			}

			column++;
		}
	}
	return weights;
}

void test_mkl_cross_validation()
{
	/* generate random data */
	index_t num=10;
	index_t dims=2;
	float64_t dist=0.5;
	SGVector<float64_t> lab(num);
	SGMatrix<float64_t> feat(dims, num);
	gen_rand_data(lab, feat, dist);

	/*create train labels */
	auto labels=std::make_shared<BinaryLabels>(lab);

	/* create train features */
	auto features=std::make_shared<DenseFeatures<float64_t>>(feat);

	/* create combined features */
	auto comb_features=std::make_shared<CombinedFeatures>();
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);
	comb_features->append_feature_obj(features);

	/* create multiple gaussian kernels */
	auto kernel=std::make_shared<CombinedKernel>();
	kernel->append_kernel(std::make_shared<GaussianKernel>(10, 0.1));
	kernel->append_kernel(std::make_shared<GaussianKernel>(10, 1));
	kernel->append_kernel(std::make_shared<GaussianKernel>(10, 2));
	kernel->init(comb_features, comb_features);

	/* create mkl using libsvm, due to a mem-bug, interleaved is not possible */
	auto svm=std::make_shared<MKLClassification>(std::make_shared<LibSVM>());
	svm->set_interleaved_optimization_enabled(false);
	svm->set_kernel(kernel);

	/* create cross-validation instance */
	index_t num_folds=3;
	auto split=std::make_shared<StratifiedCrossValidationSplitting>(labels,
			num_folds);
	auto eval=std::make_shared<ContingencyTableEvaluation>(ACCURACY);
	auto cross=std::make_shared<CrossValidation>(svm, comb_features, labels, split, eval, false);

	/* add print output listener and mkl storage listener */
	auto mkl_obs = std::make_shared<ParameterObserverCV>(true);
	cross->subscribe(&mkl_obs);

	/* perform cross-validation, this will print loads of information */
	auto result=cross->evaluate();

	/* print mkl weights */
	auto weights = calculate_weights(*mkl_obs, num_folds, 1, 3);
	weights.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	Statistics::matrix_mean(weights, false).display_vector("mean per kernel");
	Statistics::matrix_variance(weights, false).display_vector("variance per kernel");
	Statistics::matrix_std_deviation(weights, false).display_vector("std-dev per kernel");

	/* Clear */
	mkl_obs->clear();

	/* again for two runs */
	cross->set_num_runs(2);
	result=cross->evaluate();

	/* print mkl weights */
	SGMatrix<float64_t> weights_2 = calculate_weights(*mkl_obs, num_folds, 2, 3);
	weights_2.display_matrix("mkl weights");

	/* print mean and variance of each kernel weight. These could for example
	 * been used to compute confidence intervals */
	Statistics::matrix_mean(weights_2, false)
	    .display_vector("mean per kernel");
	Statistics::matrix_variance(weights_2, false)
	    .display_vector("variance per kernel");
	Statistics::matrix_std_deviation(weights_2, false)
	    .display_vector("std-dev per kernel");

	/* clean up */
}

int main()
{
//	env()->io()->set_loglevel(MSG_DEBUG);

	test_mkl_cross_validation();

	return 0;
}

