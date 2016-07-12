#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <gtest/gtest.h>
#include <shogun/labels/BinaryLabels.h>

using namespace shogun;

TEST (KNN, simple_classification)
{
	/*create a rectangle with four points as (5,5), (-5,5), (-5,-5), (5,-5)*/
	SGMatrix<float64_t> rect(2, 4);
	rect(0,0)=5;
	rect(0,1)=-5;
	rect(0,2)=-5;
	rect(0,3)=5;
	rect(1,0)=5;
	rect(1,1)=5;
	rect(1,2)=-5;
	rect(1,3)=-5;
	
	
	CMulticlassLabels* labels=new CMulticlassLabels(4);
	labels->set_label(0, 1);
	labels->set_label(1, 2);
	labels->set_label(2, 1);
	labels->set_label(3, 0);
	CDenseFeatures<float64_t>* features=new CDenseFeatures<float64_t>(rect);
	SG_REF(features);
	CEuclideanDistance* distance=new CEuclideanDistance(features, features);
	CKNN* knn=new CKNN(4, distance, labels);
    
	knn->train(features);
    
    /*Test data*/
	SGMatrix<float64_t> data(2, 1);
	data(0,0)=0;
	data(0,1)=0;
	CDenseFeatures<float64_t>* test=new CDenseFeatures<float64_t>(data);

	CMulticlassLabels* result=knn->apply_multiclass(test);

	EXPECT_EQ(1.000000, result->get_label(0));
	
	SG_UNREF(knn);
	SG_UNREF(features);

	}
