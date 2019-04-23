/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Soeren Sonnenburg, Evgeniy Andreev, 
 *          Sergey Lisitsyn
 */

#include <shogun/base/init.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/kernel/LinearKernel.h>
#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/evaluation/ContingencyTableEvaluation.h>

using namespace shogun;

void print_message(FILE* target, const char* str)
{
	fprintf(target, "%s", str);
}

void test()
{
	/* data matrix dimensions */
	index_t num_vectors=6;
	index_t num_features=2;

	/* data means -1, 1 in all components, small std deviation */
	SGVector<float64_t> mean_1(num_features);
	SGVector<float64_t> mean_2(num_features);
	SGVector<float64_t>::fill_vector(mean_1.vector, mean_1.vlen, -10.0);
	SGVector<float64_t>::fill_vector(mean_2.vector, mean_2.vlen, 10.0);
	float64_t sigma=0.5;

	SGVector<float64_t>::display_vector(mean_1.vector, mean_1.vlen, "mean 1");
	SGVector<float64_t>::display_vector(mean_2.vector, mean_2.vlen, "mean 2");

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

	SGMatrix<float64_t>::display_matrix(train_dat.matrix, train_dat.num_rows, train_dat.num_cols, "training data");

	/* training features */
	DenseFeatures<float64_t>* features=
			new DenseFeatures<float64_t>(train_dat);

	/* training labels +/- 1 for each cluster */
	SGVector<float64_t> lab(num_vectors);
	for (index_t i=0; i<num_vectors; ++i)
		lab.vector[i]=i<num_vectors/2 ? -1.0 : 1.0;

	SGVector<float64_t>::display_vector(lab.vector, lab.vlen, "training labels");

	auto labels=std::make_shared<BinaryLabels>(lab);

	/* evaluation instance */
	auto eval=std::make_shared<ContingencyTableEvaluation>(ACCURACY);

	/* kernel */
	Kernel* kernel=new LinearKernel();
	kernel->init(features, features);

	/* create svm via libsvm */
	float64_t svm_C=10;
	float64_t svm_eps=0.0001;
	CLibSVM* svm=new CLibSVM(svm_C, kernel, labels);
	svm->set_epsilon(svm_eps);

	/* now train a few times on different subsets on data and assert that
	 * results are correct (data linear separable) */

	svm->data_lock(labels, features);

	SGVector<index_t> indices(5);
	indices.vector[0]=1;
	indices.vector[1]=2;
	indices.vector[2]=3;
	indices.vector[3]=4;
	indices.vector[4]=5;
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	svm->train_locked(indices);
	BinaryLabels* output = svm->apply()->as<BinaryLabels>();
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);

	SG_SPRINT("\n\n");
	indices=SGVector<index_t>(3);
	indices.vector[0]=1;
	indices.vector[1]=2;
	indices.vector[2]=3;
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	output = svm->apply()->as<BinaryLabels>();
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);

	SG_SPRINT("\n\n");
	indices=SGVector<index_t>(4);
	indices.range_fill();
	SGVector<index_t>::display_vector(indices.vector, indices.vlen, "training indices");
	svm->train_locked(indices);
	output = svm->apply()->as<BinaryLabels>();
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "apply() output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");
	SG_SPRINT("accuracy: %f\n", eval->evaluate(output, labels));
	ASSERT(eval->evaluate(output, labels)==1);

	SG_SPRINT("normal train\n");
	svm->data_unlock();
	svm->train();
	output = svm->apply()->as<BinaryLabels>();
	ASSERT(eval->evaluate(output, labels)==1);
	SGVector<float64_t>::display_vector(output->get_labels().vector, output->get_num_labels(), "output");
	SGVector<float64_t>::display_vector(labels->get_labels().vector, labels->get_labels().vlen, "training labels");

	/* clean up */
}

int main(int argc, char **argv)
{
	init_shogun(&print_message, &print_message, &print_message);

	test();

	exit_shogun();

	return 0;
}

