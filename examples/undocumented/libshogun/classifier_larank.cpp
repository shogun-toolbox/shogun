/*
 * This software is distributed under BSD 3-clause license (see LICENSE file).
 *
 * Authors: Heiko Strathmann, Sergey Lisitsyn
 */

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/LaRank.h>

using namespace shogun;

void test()
{
	index_t num_vec=10;
	index_t num_feat=3;
	index_t num_class=num_feat; // to make data easy
	float64_t distance=15;

	// create some linearly seperable data
	SGMatrix<float64_t> matrix(num_class, num_vec);
	SGMatrix<float64_t> matrix_test(num_class, num_vec);
	MulticlassLabels* labels=new MulticlassLabels(num_vec);
	MulticlassLabels* labels_test=new MulticlassLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
	{
		index_t label=i%num_class;
		for (index_t j=0; j<num_feat; ++j)
		{
			matrix(j,i)=Math::randn_double();
			matrix_test(j,i)=Math::randn_double();
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label,i)+=distance;
		matrix_test(label,i)+=distance;
	}
	matrix.display_matrix("matrix");
	labels->get_int_labels().display_vector("labels");

	// shogun will now own the matrix created
	DenseFeatures<float64_t>* features=new DenseFeatures<float64_t>(matrix);
	DenseFeatures<float64_t>* features_test=
			new DenseFeatures<float64_t>(matrix_test);

	// create three labels
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	GaussianKernel* kernel = new GaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	CLaRank* svm = new CLaRank(10, kernel, labels);
	svm->train();
	svm->train();

	// classify on training examples
	MulticlassLabels* output=(MulticlassLabels*)svm->apply();
	output->get_labels().display_vector("batch output");

	/* assert that batch apply and apply(index_t) give same result */
	SGVector<float64_t> single_outputs(output->get_num_labels());
	for (index_t i=0; i<output->get_num_labels(); ++i)
		single_outputs[i]=svm->apply_one(i);

	single_outputs.display_vector("single_outputs");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		ASSERT(output->get_label(i)==single_outputs[i]);

	MulticlassLabels* output_test=
			(MulticlassLabels*)svm->apply(features_test);
	labels_test->get_labels().display_vector("labels_test");
	output_test->get_labels().display_vector("output_test");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		ASSERT(labels_test->get_label(i)==output_test->get_label(i));

	// free up memory
}

int main(int argc, char** argv)
{
	test();

	return 0;
}

