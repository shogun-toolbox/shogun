#include <multiclass/MulticlassLibLinear.h>
#include <features/DenseFeatures.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(MulticlassLibLinearTest,train_and_apply)
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
			matrix(j, i)=CMath::randn_double();
			matrix_test(j, i)=CMath::randn_double();
			labels->set_label(i, label);
			labels_test->set_label(i, label);
		}

		/* make sure data is linearly seperable per class */
		matrix(label, i)+=distance;
		matrix_test(label, i)+=distance;
	}
	//matrix.display_matrix("matrix");
	//labels->get_int_labels().display_vector("labels");

	// shogun will now own the matrix created
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(matrix);
	CDenseFeatures<float64_t>* features_test=new CDenseFeatures<float64_t>(
			matrix_test);


	float64_t C=1.0;

	CMulticlassLibLinear* mocas=new CMulticlassLibLinear(C, features,
			labels);
	mocas->parallel->set_num_threads(1);
	mocas->set_epsilon(1e-5);
	mocas->train();

	CLabels* pred=mocas->apply(features_test);
	for (int i=0; i<features_test->get_num_vectors(); ++i)
		EXPECT_EQ(labels_test->get_label(i), ((CMulticlassLabels* )pred)->get_label(i));

	SG_UNREF(mocas);
	SG_UNREF(labels_test);
	SG_UNREF(pred);
}
