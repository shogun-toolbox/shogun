#include <gtest/gtest.h>

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DataGenerator.h>

using namespace shogun;

void generate_knn_data(SGMatrix<float64_t>& feat, SGVector<float64_t>& lab,
	   	int32_t num, int32_t classes, int32_t feats)
{
	CMath::init_random(1);
	feat = CDataGenerator::generate_gaussians(num,classes,feats);
	for( int32_t i = 0 ; i < classes ; ++i )
		for( int32_t j = 0 ; j < num ; ++j )
			lab[i*num+j] = double(i);

}

class KNNTest : public ::testing::Test
{
protected:
	virtual void SetUp()
	{
		SGVector<float64_t> lab(classes*num);
		SGMatrix<float64_t> feat(feats, classes*num);

		generate_knn_data(feat, lab, num, classes, feats);

		train = SGVector<index_t>(index_t(num*classes*0.75));
		test = SGVector<index_t>(index_t(num*classes*0.25));

		//generate random subset for train and test data
		train.random(0, classes*num-1);
		test.random(0, classes*num-1);

		labels = new CMulticlassLabels(lab);
		SG_REF(labels);
		features = new CDenseFeatures<float64_t>(feat);
		SG_REF(features);
		features_test = (CDenseFeatures<float64_t>*)features->clone();
		labels_test = (CLabels*)labels->clone();

		features->add_subset(train);
		labels->add_subset(train);
		features_test->add_subset(test);
		labels_test->add_subset(test);

		distance = new CEuclideanDistance();
		SG_REF(distance);
	}

	virtual void TearDown()
	{
		SG_UNREF(features);
		SG_UNREF(features_test);
		SG_UNREF(labels);
		SG_UNREF(labels_test);
		SG_UNREF(distance);
	}

	const int32_t k = 4;
	const index_t num = 50;
	const index_t feats = 2;
	const int32_t classes = 3;

	SGVector<index_t> train;
	SGVector<index_t> test;

	CMulticlassLabels* labels;
	CLabels* labels_test;
	CDenseFeatures<float64_t>* features;
	CDenseFeatures<float64_t>* features_test;
	CDistance* distance;
};

// FIXME: templated tests but it doesn't work with enums :()
// typedef ::testing::Types<KNN_BRUTE, KNN_KDTREE, KNN_LSH> KNNTypes;
TEST_F(KNNTest, brute_solver)
{
	auto knn = some<CKNN>(k, distance, labels, KNN_BRUTE);
	knn->train(features);
	auto output = knn->apply(features_test)->as<CMulticlassLabels>();
	SG_REF(output);

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
}

TEST_F(KNNTest, kdtree_solver)
{
	auto knn = some<CKNN>(k, distance, labels, KNN_KDTREE);
	knn->train(features);
	auto output = knn->apply(features_test)->as<CMulticlassLabels>();
	SG_REF(output);

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
}

TEST_F(KNNTest, lsh_solver)
{
	auto knn = some<CKNN>(k, distance, labels, KNN_LSH);
	knn->train(features);
	auto output = knn->apply(features_test)->as<CMulticlassLabels>();
	SG_REF(output);

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
}

TEST_F(KNNTest, lsh_solver_sparse)
{
	auto knn = some<CKNN>(k, distance, labels, KNN_LSH);
	// TODO: the sparse features should be actually sparse
	auto features_sparse = new CSparseFeatures<float64_t>(features);
	auto features_test_sparse = new CSparseFeatures<float64_t>(features_test);
	knn->train(features_sparse);
	auto output = knn->apply(features_test_sparse)->as<CMulticlassLabels>();
	SG_REF(output);

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), ((CMulticlassLabels*)labels_test)->get_label(i));

	SG_UNREF(output);
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
