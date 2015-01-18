#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/multiclass/LaRank.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LaRank,train)
{
	index_t num_vec=10;
	index_t num_feat=3;
	index_t num_class=num_feat; // to make data easy
	float64_t distance=15;

	// create some linearly seperable data
	SGMatrix<float64_t> matrix(num_class, num_vec);
	SGMatrix<float64_t> matrix_test(num_class, num_vec);
	CMulticlassLabels* labels=new CMulticlassLabels(num_vec);
	CMulticlassLabels* labels_test=new CMulticlassLabels(num_vec);
	for (index_t i=0; i<num_vec; ++i)
	{
		index_t label=i%num_class;
		for (index_t j=0; j<num_feat; ++j)
		{
			matrix(j,i)=CMath::randn_double();
			matrix_test(j,i)=CMath::randn_double();
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label,i)+=distance;
		matrix_test(label,i)+=distance;
	}
	//matrix.display_matrix("matrix");
	//labels->get_int_labels().display_vector("labels");

	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);
	CDenseFeatures<float64_t>* features_test=
			new CDenseFeatures<float64_t>(matrix_test);

	// create three labels
	for (index_t i=0; i<num_vec; ++i)
		labels->set_label(i, i%num_class);

	// create gaussian kernel with cache 10MB, width 0.5
	CGaussianKernel* kernel = new CGaussianKernel(10, 0.5);
	kernel->init(features, features);

	// create libsvm with C=10 and train
	CLaRank* svm = new CLaRank(10, kernel, labels);
	svm->train();
	svm->train();

	// classify on training examples
	CMulticlassLabels* output=(CMulticlassLabels*)svm->apply();
	//output->get_labels().display_vector("batch output");

	/* assert that batch apply and apply(index_t) give same result */
	SGVector<float64_t> single_outputs(output->get_num_labels());
	for (index_t i=0; i<output->get_num_labels(); ++i)
		single_outputs[i]=svm->apply_one(i);

	//single_outputs.display_vector("single_outputs");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		EXPECT_EQ(output->get_label(i), single_outputs[i]);

	// predict test labels (since data is easy this has to be correct
	CMulticlassLabels* output_test=
			(CMulticlassLabels*)svm->apply(features_test);
	//labels_test->get_labels().display_vector("labels_test");
	//output_test->get_labels().display_vector("output_test");

	for (index_t i=0; i<output->get_num_labels(); ++i)
		EXPECT_EQ(labels_test->get_label(i), output_test->get_label(i));

	// free up memory
	SG_UNREF(output);
	SG_UNREF(labels_test);
	SG_UNREF(output_test);
	SG_UNREF(svm);
}

