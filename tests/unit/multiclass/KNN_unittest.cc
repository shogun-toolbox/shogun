#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <gtest/gtest.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

#ifdef HAVE_LAPACK
void generate_knn_data(SGMatrix<float64_t>& feat, SGVector<float64_t>& lab,
	   	int32_t num, int32_t classes, int32_t feats)
{
	CMath::init_random(1);
	feat = CDataGenerator::generate_gaussians(num,classes,feats);
	for( int i = 0 ; i < classes ; ++i )
		for( int j = 0 ; j < num ; ++j )
			lab[i*num+j] = double(i);

}

TEST(KNN, brute_solver)
{
	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_knn_data(feat, lab, num, classes, feats);
	
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));

	//generate random subset for train and test data
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();	
	CLabels* labels_test = (CLabels*) labels->clone();

	int32_t k=4;
	CEuclideanDistance* distance = new CEuclideanDistance();	
	CKNN* knn=new CKNN (k, distance, labels, KNN_BRUTE);
	SG_REF(knn);

	features->add_subset(train);
	labels->add_subset(train);	
	knn->train(features);

	features_test->add_subset(test);
	labels_test->add_subset(test);
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(knn->apply(features_test));
	SG_REF(output);
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
	SG_UNREF(knn);
	SG_UNREF(features_test);
	SG_UNREF(labels_test);
}

TEST(KNN, kdtree_solver)
{

	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);
	
	generate_knn_data(feat, lab, num, classes, feats);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();	
	CLabels* labels_test = (CLabels*) labels->clone();

	int32_t k=4;
	CEuclideanDistance* distance = new CEuclideanDistance();	
	CKNN* knn=new CKNN (k, distance, labels, KNN_KDTREE);
	SG_REF(knn);

	features->add_subset(train);
	labels->add_subset(train);	
	knn->train(features);

	features_test->add_subset(test);
	labels_test->add_subset(test);
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(knn->apply(features_test));
	SG_REF(output);
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
	SG_UNREF(knn);
	SG_UNREF(features_test);
	SG_UNREF(labels_test);
}


TEST(KNN, lsh_solver)
{

	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);
	
	generate_knn_data(feat, lab, num, classes, feats);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();	
	CLabels* labels_test = (CLabels*) labels->clone();

	int32_t k=4;
	CEuclideanDistance* distance = new CEuclideanDistance();	
	CKNN* knn=new CKNN (k, distance, labels, KNN_LSH);
	SG_REF(knn);

	features->add_subset(train);
	labels->add_subset(train);	
	knn->train(features);

	features_test->add_subset(test);
	labels_test->add_subset(test);
	CMulticlassLabels* output=CLabelsFactory::to_multiclass(knn->apply(features_test));
	SG_REF(output);
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(features_test);
	SG_UNREF(labels_test);
	SG_UNREF(output);
	SG_UNREF(knn);
}

TEST(KNN, classify_multiple_brute)
{
	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_knn_data(feat, lab, num, classes, feats);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();  
	CLabels* labels_test = (CLabels*) labels->clone();

	int32_t k=4;
	CEuclideanDistance* distance = new CEuclideanDistance();                
	CKNN* knn=new CKNN (k, distance, labels, KNN_BRUTE);
	SG_REF(knn);

	features->add_subset(train);
	labels->add_subset(train);
	knn->train(features);

	// classify for multiple k
	features_test->add_subset(test);
	labels_test->add_subset(test);

	CEuclideanDistance* dist = new CEuclideanDistance(features, ((CDotFeatures*)features_test));  
	knn->set_distance(dist);
	SGMatrix<int32_t> out_mat =knn->classify_for_multiple_k();
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		for ( index_t j = 0; j < k; ++j )
			EXPECT_EQ(out_mat(i, j), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(knn);
	SG_UNREF(features_test);
	SG_UNREF(labels_test);
}


TEST(KNN, classify_multiple_kdtree)
{

	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_knn_data(feat, lab, num, classes, feats);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	train.random(0, classes*num-1);
	test.random(0, classes*num-1);

	CMulticlassLabels* labels = new CMulticlassLabels(lab);
	CDenseFeatures< float64_t >* features = new CDenseFeatures< float64_t >(feat);
	CFeatures* features_test = (CFeatures*) features->clone();	
	CLabels* labels_test = (CLabels*) labels->clone();

	int32_t k=4;
	CEuclideanDistance* distance = new CEuclideanDistance();	
	CKNN* knn=new CKNN (k, distance, labels, KNN_KDTREE);
	SG_REF(knn);

	features->add_subset(train);
	labels->add_subset(train);
	knn->train(features);

	// classify for multiple k
	features_test->add_subset(test);
	labels_test->add_subset(test);
	CEuclideanDistance* dist = new CEuclideanDistance(features, ((CDotFeatures*)features_test));
	knn->set_distance(dist);
	SGMatrix<int32_t> out_mat =knn->classify_for_multiple_k();
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		for ( index_t j = 0; j < k; ++j )
			EXPECT_EQ(out_mat(i, j), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(knn);
	SG_UNREF(features_test);
	SG_UNREF(labels_test);
}
#endif /* HAVE_LAPACK */

