/*
 * Copyright (c) The Shogun Machine Learning Toolbox
 * Written (w) 2016 Saurabh Mahindre
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 *    list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
 * ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * The views and conclusions contained in the software and documentation are those
 * of the authors and should not be interpreted as representing official policies,
 * either expressed or implied, of the Shogun Development Team.
 *
 */

#include <gtest/gtest.h>
#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/evaluation/MulticlassAccuracy.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/mathematics/NormalDistribution.h>

#include <random>

using namespace shogun;

template <typename PRNG>
void generate_data(SGMatrix<float64_t>& mat, SGVector<float64_t> &lab, PRNG& prng)
{
	int32_t num=lab.size();
	NormalDistribution<float64_t> normal_dist;
	for (index_t i=0; i<num; i++)
	{
		mat(0,i)=i<num/2 ? 0+(normal_dist(prng)*4) : 100+(normal_dist(prng)*4)	;
		mat(1,i)=i;
	}

	for (index_t i=0; i<num; ++i)
		lab.vector[i]=i<num/2 ? 0 : 1;

}

TEST(CrossValidation_multithread, LibSVM_unlocked)
{
	int32_t seed = 100;
	int32_t num=100;
	SGMatrix<float64_t> mat(2, num);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num);

	/*create simple linearly separable data*/
	std::mt19937_64 prng(seed);
	generate_data(mat, lab, prng);

	for (index_t i=0; i<num/2; ++i)
		lab.vector[i]-=1;
	auto labels=std::make_shared<BinaryLabels>(lab);

	auto features=
			std::make_shared<DenseFeatures<float64_t>>(mat);


	int32_t width=100;
	auto kernel=std::make_shared<GaussianKernel>(width);
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=1;
	auto svm=std::make_shared<LibSVM>(svm_C, kernel, labels);

	auto eval_crit=
			std::make_shared<ContingencyTableEvaluation>(ACCURACY);

	index_t n_folds=4;
	auto splitting=
			std::make_shared<StratifiedCrossValidationSplitting>(labels, n_folds);
	splitting->put("seed", seed);

	auto cross=std::make_shared<CrossValidation>(svm, features, labels,
			splitting, eval_crit);

	cross->set_autolock(false);
	cross->set_num_runs(4);
	cross->parallel->set_num_threads(1);

	auto result1=cross->evaluate()->as<CrossValidationResult>();;
	float64_t mean1 = result1->get_mean();

	cross->parallel->set_num_threads(3);

	auto result2=cross->evaluate()->as<CrossValidationResult>();;
	float64_t mean2 = result2->get_mean();

	EXPECT_EQ(mean1, mean2);

	/* clean up */




}

TEST(CrossValidation_multithread, KNN)
{
	int32_t seed = 100;
	int32_t num=100;
	SGMatrix<float64_t> mat(2, num);

	SGVector<float64_t> lab(num);

	/*create simple linearly separable data*/
	std::mt19937_64 prng(seed);
	generate_data(mat, lab, prng);
	auto labels=std::make_shared<MulticlassLabels>(lab);

	auto features=
			std::make_shared<DenseFeatures<float64_t>>(mat);

	/* create knn */
	auto distance = std::make_shared<EuclideanDistance>(features, features);
	auto knn=std::make_shared<KNN>(4, distance, labels);
	/* evaluation criterion */
	auto eval_crit = std::make_shared<MulticlassAccuracy> ();

	/* splitting strategy */
	index_t n_folds=4;
	auto splitting=
			std::make_shared<StratifiedCrossValidationSplitting>(labels, n_folds);
	splitting->put("seed", seed);

	auto cross=std::make_shared<CrossValidation>(knn, features, labels,
			splitting, eval_crit);

	cross->set_autolock(false);
	cross->set_num_runs(4);
	cross->parallel->set_num_threads(1);

	auto result1=cross->evaluate()->as<CrossValidationResult>();
	float64_t mean1 = result1->get_mean();

	cross->parallel->set_num_threads(3);

	auto result2=cross->evaluate()->as<CrossValidationResult>();
	float64_t mean2 = result2->get_mean();

	EXPECT_EQ(mean1, mean2);





}
