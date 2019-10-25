/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Evgeniy Andreev, Soumyajit De, 
 *          Jacob Walker, Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/classifier/svm/SVMLight.h>
#include <shogun/evaluation/CrossValidation.h>
#include <shogun/evaluation/StratifiedCrossValidationSplitting.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>
#include <shogun/lib/Time.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test_cross_validation()
{
	/* data matrix dimensions */
	index_t num_vectors=50;
	index_t num_features=5;

	/* data means -1, 1 in all components, std deviation of sigma */
	SGVector<float64_t> mean_1(num_features);
	SGVector<float64_t> mean_2(num_features);
	SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -1.0);
	SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 1.0);
	float64_t sigma=1.5;

	/* fill data matrix around mean */
	SGMatrix<float64_t> train_dat(num_features, num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
	{
		for (index_t j=0; j<num_features; ++j)
		{
			float64_t mean=i<num_vectors/2 ? mean_1.vector[0] : mean_2.vector[0];
			train_dat.matrix[i*num_features+j]=Math::normal_random(mean, sigma);
		}
	}

	/* training features */
	auto features=std::make_shared<DenseFeatures<float64_t>>(train_dat);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
		lab.vector[i]=i<num_vectors/2 ? -1.0 : 1.0;

	auto labels=std::make_shared<BinaryLabels>(lab);

	/* gaussian kernel */
	auto kernel=std::make_shared<GaussianKernel>();
	kernel->set_width(10);
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=1;
	float64_t svm_eps=0.0001;
	auto svm=std::make_shared<LibSVM>(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	/* train and output the normal way */
	SG_SPRINT("starting normal training\n");
	svm->train(features);
	auto output = svm->apply(features)->as<BinaryLabels>();

	/* evaluation criterion */
	auto eval_crit=
			std::make_shared<ContingencyTableEvaluation>(ACCURACY);

	/* evaluate training error */
	float64_t eval_result=eval_crit->evaluate(output, labels);
	SG_SPRINT("training accuracy: %f\n", eval_result);

	/* assert that regression "works". this is not guaranteed to always work
	 * but should be a really coarse check to see if everything is going
	 * approx. right */
	ASSERT(eval_result<2);

	/* splitting strategy */
	index_t n_folds=3;
	auto splitting=
			std::make_shared<StratifiedCrossValidationSplitting>(labels, n_folds);

	/* cross validation instance, 10 runs, 95% confidence interval */
	auto cross=std::make_shared<CrossValidation>(svm, features, labels,
			splitting, eval_crit);

	cross->set_num_runs(5);
//	cross->set_conf_int_alpha(0.05);

	std::shared_ptr<CrossValidationResult> tmp;
	/* no locking */
	index_t repetitions=5;
	SG_SPRINT("unlocked x-val\n");
	kernel->init(features, features);
	cross->set_autolock(false);
	Time time;
	time.start();
	for (index_t i=0; i<repetitions; ++i)
	{
		tmp = cross->evaluate()->as<CrossValidationResult>();
	}

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* auto_locking in every iteration of this loop (better, not so nice) */
	SG_SPRINT("locked in every iteration x-val\n");
	cross->set_autolock(true);
	time.start();

	for (index_t i=0; i<repetitions; ++i)
        {
                tmp = cross->evaluate()->as<CrossValidationResult>();
        }

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* lock once before, (no locking/unlocking in this loop) */
	svm->data_lock(labels, features);
	SG_SPRINT("locked x-val\n");
	time.start();

        for (index_t i=0; i<repetitions; ++i)
        {
                tmp = cross->evaluate()->as<CrossValidationResult>();
        }

	time.stop();
	SG_SPRINT("%f sec\n", time.cur_time_diff());

	/* clean up */
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test_cross_validation();

	exit_shogun();

	return 0;
}

