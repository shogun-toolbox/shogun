#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/lib/SGMatrix.h>

#include <gtest/gtest.h>

using ::testing::Test;
using namespace shogun;

// Results compared with sklearn
// https://gist.github.com/micmn/f93f723b74db2a1eb5875f63d841bdc1
const index_t num_vectors = 5;
const index_t num_features = 3;
const index_t target_dim = 2;

const float64_t train_data[] = {1, 1, 1, 1, 2, 3, 5, 6, 1, 2, 2, 2, 1, 1, 1};
const float64_t test_data[] = {3, 3, 3, 7, 4, 1};
const float64_t resdata[] = {-0.17645841, 0.013962, -0.16082441, 0.03640145};

template <template <typename> class Container>
void load_data(SGMatrix<float64_t>& train, Container<float64_t>& test)
{
	for (auto i = 0; i < train.size(); ++i)
		train[i] = train_data[i];

	for (auto i = 0; i < test.size(); ++i)
		test[i] = test_data[i];
}

TEST(KernelPCA, apply)
{
	index_t num_test_vectors = 2;

	SGMatrix<float64_t> train_matrix(num_features, num_vectors);
	SGMatrix<float64_t> test_matrix(num_features, num_test_vectors);
	load_data(train_matrix, test_matrix);

	CDenseFeatures<float64_t>* train_feats =
	    new CDenseFeatures<float64_t>(train_matrix);

	CDenseFeatures<float64_t>* test_feats =
	    new CDenseFeatures<float64_t>(test_matrix);

	SG_REF(train_feats)
	SG_REF(test_feats)

	CGaussianKernel* kernel = new CGaussianKernel();
	SG_REF(kernel)
	kernel->set_width(1);

	CKernelPCA* kpca = new CKernelPCA(kernel);
	SG_REF(kpca)
	kpca->set_target_dim(target_dim);
	kpca->fit(train_feats);

	SGMatrix<float64_t> embedding = kpca->apply(test_feats)
	                                    ->as<CDenseFeatures<float64_t>>()
	                                    ->get_feature_matrix();

	// allow embedding with opposite sign
	for (index_t i = 0; i < num_test_vectors * target_dim; ++i)
		EXPECT_NEAR(CMath::abs(embedding[i]), CMath::abs(resdata[i]), 1E-6);

	SG_UNREF(train_feats)
	SG_UNREF(test_feats)
	SG_UNREF(kpca);
	SG_UNREF(kernel);
}

TEST(KernelPCA, apply_to_feature_vector)
{
	SGMatrix<float64_t> train_matrix(num_features, num_vectors);
	SGVector<float64_t> test_vector(num_features);
	load_data(train_matrix, test_vector);

	CDenseFeatures<float64_t>* train_feats =
	    new CDenseFeatures<float64_t>(train_matrix);
	SG_REF(train_feats)

	CGaussianKernel* kernel = new CGaussianKernel();
	SG_REF(kernel)
	kernel->set_width(1);

	CKernelPCA* kpca = new CKernelPCA(kernel);
	SG_REF(kpca)
	kpca->set_target_dim(target_dim);
	kpca->fit(train_feats);

	SGVector<float64_t> embedding = kpca->apply_to_feature_vector(test_vector);

	// allow embedding with opposite sign
	for (index_t i = 0; i < target_dim; ++i)
		EXPECT_NEAR(CMath::abs(embedding[i]), CMath::abs(resdata[i]), 1E-6);

	SG_UNREF(train_feats)
	SG_UNREF(kpca);
	SG_UNREF(kernel);
}
