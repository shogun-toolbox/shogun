#include <shogun/features/DenseFeatures.h>
#include <shogun/labels/MulticlassLabels.h>
#include <shogun/multiclass/QDA.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/Statistics.h>
#include <gtest/gtest.h>

#define	NUM  20
#define DIMS 2
#define CLASSES 2
#define ALPHA1 1
#define ALPHA2 2
#define D 15

using namespace shogun;

TEST(QDA,train_test)
{
	SGMatrix< float64_t > vec_train(DIMS, CLASSES*NUM);
	
	CMulticlassLabels* lab_train = new CMulticlassLabels(CLASSES*NUM);
	CMulticlassLabels* lab_test = new CMulticlassLabels(CLASSES*NUM);

	SGVector<float64_t> mean1(DIMS);
	SGVector<float64_t> mean2(DIMS);
	for( int i = 0; i < DIMS; ++i )
	{
		mean1[i] = 1;
		mean2[i] = -1;
	}

	// covariance matrix is of the form of diagonal matrix, specfically, ALPHA*I
	SGMatrix<float64_t> cov1(DIMS, DIMS);
	SGMatrix<float64_t> cov2(DIMS, DIMS);
	for( int i = 0; i < DIMS; ++i )
		for( int j = 0; j < DIMS; ++j )
			if (i == j)
			{
				cov1(i,j) = ALPHA1*1;
				cov2(i,j) = ALPHA2*2;
			}
			else
			{
				cov1(i,j) = 0;
				cov2(i,j) = 0;
			}

	SGMatrix< float64_t > vec_train1 = CStatistics::sample_from_gaussian(mean1, cov1, NUM);
	SGMatrix< float64_t > vec_train2 = CStatistics::sample_from_gaussian(mean2, cov2, NUM);
	
	//Adding and subtracting D (distance) to ensure that they are separable
	for( int i = 0; i < DIMS; ++i )
		for( int j = 0; j < CLASSES*NUM; ++j )
		{
			vec_train(i,j) = vec_train1(i,j) + D;
			if (j >= NUM)
				vec_train(i,j) = vec_train2(i,j-NUM) - D;
		}
	
	for( int i = 0; i < NUM*CLASSES; ++i )
		if (i >=NUM)
			lab_train->set_label(i,1);
		else
			lab_train->set_label(i,0);
	
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
