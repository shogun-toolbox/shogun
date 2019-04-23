#include <gtest/gtest.h>

#include <shogun/labels/MulticlassLabels.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/SparseFeatures.h>
#include <shogun/multiclass/KNN.h>
#include <shogun/distance/EuclideanDistance.h>
#include <shogun/labels/BinaryLabels.h>
#include <shogun/features/DataGenerator.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

template <typename PRNG>
void generate_knn_data(SGMatrix<float64_t>& feat, SGVector<float64_t>& lab,
	   	int32_t num, int32_t classes, int32_t feats, PRNG& prng)
{
	feat = DataGenerator::generate_gaussians(num,classes,feats, prng);
	for( int32_t i = 0 ; i < classes ; ++i )
		for( int32_t j = 0 ; j < num ; ++j )
			lab[i*num+j] = double(i);

}

class KNNTest : public ::testing::Test
{
protected:
	virtual void SetUp()
	{
		std::mt19937_64 prng(37);

		SGVector<float64_t> lab(classes*num);
		SGMatrix<float64_t> feat(feats, classes*num);

		generate_knn_data(feat, lab, num, classes, feats, prng);

		train = SGVector<index_t>(index_t(num*classes*0.75));
		test = SGVector<index_t>(index_t(num*classes*0.25));

		//generate random subset for train and test data
		random::fill_array(train, 0, classes*num-1, prng);
		random::fill_array(test, 0, classes*num-1, prng);

		labels = std::make_shared<MulticlassLabels>(lab);

		features = std::make_shared<DenseFeatures<float64_t>>(feat);

		features_test = features->clone()->as<DenseFeatures<float64_t>>();
		labels_test = labels->clone()->as<MulticlassLabels>();

		features->add_subset(train);
		labels->add_subset(train);
		features_test->add_subset(test);
		labels_test->add_subset(test);

		distance = std::make_shared<EuclideanDistance>();

	}

	virtual void TearDown()
	{





	}

	const int32_t k = 4;
	const index_t num = 50;
	const index_t feats = 2;
	const int32_t classes = 3;

	SGVector<index_t> train;
	SGVector<index_t> test;

	std::shared_ptr<MulticlassLabels> labels;
	std::shared_ptr<MulticlassLabels> labels_test;
	std::shared_ptr<DenseFeatures<float64_t>> features;
	std::shared_ptr<DenseFeatures<float64_t>> features_test;
	std::shared_ptr<Distance> distance;
};

// FIXME: templated tests but it doesn't work with enums :()
// typedef ::testing::Types<KNN_BRUTE, KNN_KDTREE, KNN_LSH> KNNTypes;
TEST_F(KNNTest, brute_solver)
{
	auto knn = std::make_shared<KNN>(k, distance, labels, KNN_BRUTE);
	knn->train(features);
	auto output = knn->apply(features_test)->as<MulticlassLabels>();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), labels_test->get_label(i));
}

TEST_F(KNNTest, kdtree_solver)
{
	auto knn = std::make_shared<KNN>(k, distance, labels, KNN_KDTREE);
	knn->train(features);
	auto output = knn->apply(features_test)->as<MulticlassLabels>();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), labels_test->get_label(i));

}

TEST_F(KNNTest, lsh_solver)
{
	auto knn = std::make_shared<KNN>(k, distance, labels, KNN_LSH);
	knn->train(features);
	auto output = knn->apply(features_test)->as<MulticlassLabels>();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), labels_test->get_label(i));

}

TEST_F(KNNTest, lsh_solver_sparse)
{
	auto knn = std::make_shared<KNN>(k, distance, labels, KNN_LSH);
	// TODO: the sparse features should be actually sparse
	auto features_sparse = std::make_shared<SparseFeatures<float64_t>>(features);
	auto features_test_sparse = std::make_shared<SparseFeatures<float64_t>>(features_test);
	knn->train(features_sparse);
	auto output = knn->apply(features_test_sparse)->as<MulticlassLabels>();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		EXPECT_EQ(output->get_label(i), labels_test->get_label(i));

}

TEST(KNN, classify_multiple_brute)
{
	std::mt19937_64 prng(17);

	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_knn_data(feat, lab, num, classes, feats, prng);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	random::fill_array(train, 0, classes*num-1, prng);
	random::fill_array(test, 0, classes*num-1, prng);

	auto labels = std::make_shared<MulticlassLabels>(lab);
	auto features = std::make_shared<DenseFeatures< float64_t >>(feat);
	auto features_test = features->clone()->as<DotFeatures>();
	auto labels_test = labels->clone()->as<MulticlassLabels>();

	int32_t k=4;
	auto distance = std::make_shared<EuclideanDistance>();
	auto knn=std::make_shared<KNN> (k, distance, labels, KNN_BRUTE);


	features->add_subset(train);
	labels->add_subset(train);
	knn->train(features);

	// classify for multiple k
	features_test->add_subset(test);
	labels_test->add_subset(test);

	auto dist = std::make_shared<EuclideanDistance>(features, features_test);
	knn->set_distance(dist);
	SGMatrix<int32_t> out_mat =knn->classify_for_multiple_k();
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		for ( index_t j = 0; j < k; ++j )
			EXPECT_EQ(out_mat(i, j), labels_test->get_label(i));




}


TEST(KNN, classify_multiple_kdtree)
{
	std::mt19937_64 prng(17);

	int32_t num = 50;
	int32_t feats = 2;
	int32_t classes = 3;

	SGVector< float64_t > lab(classes*num);
	SGMatrix< float64_t > feat(feats, classes*num);

	generate_knn_data(feat, lab, num, classes, feats, prng);
	SGVector<index_t> train (int32_t(num*classes*0.75));
	SGVector<index_t> test (int32_t(num*classes*0.25));
	random::fill_array(train, 0, classes*num-1, prng);
	random::fill_array(test, 0, classes*num-1, prng);

	auto labels = std::make_shared<MulticlassLabels>(lab);
	auto features = std::make_shared<DenseFeatures< float64_t >>(feat);
	auto features_test = features->clone()->as<DotFeatures>();
	auto labels_test = labels->clone()->as<MulticlassLabels>();

	int32_t k=4;
	auto distance = std::make_shared<EuclideanDistance>();
	auto knn=std::make_shared<KNN>(k, distance, labels, KNN_KDTREE);


	features->add_subset(train);
	labels->add_subset(train);
	knn->train(features);

	// classify for multiple k
	features_test->add_subset(test);
	labels_test->add_subset(test);
	auto dist = std::make_shared<EuclideanDistance>(features, features_test);
	knn->set_distance(dist);
	SGMatrix<int32_t> out_mat =knn->classify_for_multiple_k();
	features_test->remove_subset();

	for ( index_t i = 0; i < labels_test->get_num_labels(); ++i )
		for ( index_t j = 0; j < k; ++j )
			EXPECT_EQ(out_mat(i, j), labels_test->get_label(i));




}
