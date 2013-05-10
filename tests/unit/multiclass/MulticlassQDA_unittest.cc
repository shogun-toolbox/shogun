#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/QDA.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

#define d 15
#define num_classes 2

using namespace shogun;

TEST(QDA,train_test)
{
	index_t num_vectors = 20;
	index_t num_feat = 2;
	
	SGMatrix<float64_t> vec_train(num_feat,num_vectors);
	CMulticlassLabels* lab_train = new CMulticlassLabels(num_vectors);

	CMulticlassLabels* lab_test = new CMulticlassLabels(num_vectors);

	for (index_t i = 0; i < num_vectors; i++)
	{
		for (index_t j = 0; j < num_feat; j++)
		{
			vec_train(j,i) = CMath::random(-1.0,1.0);
		}
		if(CMath::pow(vec_train(0,i),2) < vec_train(1,i) )
		{
			//vec_train((0,i),2) = vec_train((0,i),2) + d;
			vec_train(1,i) = vec_train(1,i) + d;
			lab_train->set_label(i, 1);
		}
		else 
		{
			//vec_train((0,i),2) = vec_train((0,i),2) - d;
			vec_train(1,i) = vec_train(1,i) - d;
			lab_train->set_label(i, 0);
		}
	}
	
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(vec_train);

	CQDA* qda = new CQDA(features, lab_train);
	SG_REF(qda);
	qda->train();
	
	CDenseFeatures<float64_t>* test_features=new CDenseFeatures<float64_t>(vec_train);
	lab_test = CMulticlassLabels::obtain_from_generic(qda->apply(test_features));
	
	for (index_t i = 0; i < num_vectors; i++)
		EXPECT_EQ(lab_test->get_label(i),lab_train->get_label(i));
	
	SG_UNREF(lab_test);
	SG_UNREF(qda);
}
