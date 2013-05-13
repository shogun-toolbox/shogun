#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/QDA.h>
#include <shogun/mathematics/Math.h>
#include <shogun/features/DataGenerator.h>
#include <gtest/gtest.h>

#define	NUM  50
#define DIMS 2
#define CLASSES 2

using namespace shogun;

TEST(QDA,train_test)
{
	SGMatrix< float64_t > vec_train(DIMS, CLASSES*NUM);
	SGVector< float64_t > lab(CLASSES*NUM);

	CMulticlassLabels* lab_test = new CMulticlassLabels(CLASSES*NUM);

	vec_train = CDataGenerator::generate_gaussians(NUM,CLASSES,DIMS);

	for( int i = 0; i < CLASSES; ++i )
		for( int j = 0; j < NUM; ++j )
			lab[i*NUM+j] = double(i);

	CMulticlassLabels* lab_train = new CMulticlassLabels(lab);
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(vec_train);

	CQDA* qda = new CQDA(features, lab_train);
	SG_REF(qda);
	qda->train();
	
	CDenseFeatures<float64_t>* test_features=new CDenseFeatures<float64_t>(vec_train);
	lab_test = CMulticlassLabels::obtain_from_generic(qda->apply(test_features));
	
	for ( int i = 0; i < CLASSES*NUM; ++i )
		EXPECT_EQ(lab_test->get_label(i),lab_train->get_label(i));
	
	SG_UNREF(lab_test);
	SG_UNREF(qda);
}
