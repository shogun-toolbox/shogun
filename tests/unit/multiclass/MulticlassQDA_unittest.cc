#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/QDA.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <gtest/gtest.h>

#define	NUM  20
#define DIMS 2
#define CLASSES 2
#define D 15

using namespace shogun;

TEST(QDA,train_test)
{
	SGMatrix< float64_t > vec_train(DIMS, CLASSES*NUM);
	
	CMulticlassLabels* lab_train = new CMulticlassLabels(CLASSES*NUM);
	CMulticlassLabels* lab_test = new CMulticlassLabels(CLASSES*NUM);

	for (index_t i = 0; i < NUM*CLASSES; i++)
	{
		for (index_t j = 0; j < DIMS; j++)
		{
			vec_train(j,i) = CMath::random(-1.0,1.0);
		}
		if(CMath::pow(vec_train(0,i),2) < vec_train(1,i) )
		{
			vec_train(1,i) = vec_train(1,i) + D;
			lab_train->set_label(i, 1);
		}
		else 
		{
			vec_train(1,i) = vec_train(1,i) - D;
			lab_train->set_label(i, 0);
		}
	}
	
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(vec_train);

	CQDA* qda = new CQDA(features, lab_train);
	SG_REF(qda);
	qda->train();
	
	CDenseFeatures<float64_t>* test_features=new CDenseFeatures<float64_t>(vec_train);
	
	lab_test = CLabelsFactory::to_multiclass(qda->apply(test_features));
    SG_REF(lab_test);

	for ( int i = 0; i < CLASSES*NUM; ++i )
		EXPECT_EQ(lab_test->get_label(i),lab_train->get_label(i));
	
	SG_UNREF(lab_test);
	SG_UNREF(qda);
}
