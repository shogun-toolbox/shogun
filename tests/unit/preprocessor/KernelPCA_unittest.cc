#include <shogun/features/Features.h>
#include <shogun/preprocessor/KernelPCA.h>
#include <shogun/lib/tapkee/tapkee_shogun.hpp>
#include <shogun/kernel/Kernel.h>

#include <shogun/kernel/GaussianKernel.h>
#include <iostream>
#include <shogun/lib/SGMatrix.h>
#include <shogun/features/Features.h>

#include <shogun/base/init.h>
#include <gtest/gtest.h>
#include <gmock/gmock.h>

using ::testing::Test;
using namespace shogun;

#ifdef HAVE_LAPACK
TEST(KernelPCA, DISABLED_apply_to_feature_matrix_input)
{
	float64_t data[] = {1, 1, 1,
                      1, 2, 3,
                      5, 6, 1,
                      2, 2, 2,
                      1, 1, 1};
	float64_t resdata[] = {-1.526879008202007e-02,  6.902776989923266e-01,
                         -4.032822763552926e-01, -5.151523890814317e-01,
                         8.444041004961732e-01, -4.318711485273607e-01,
                         -4.105842439768400e-01, -4.335318603758601e-01,
                         -1.526879008202015e-02, 6.902776989923268e-01
	                        };// column-wise
	int32_t num_vectors = 5;
	int32_t num_features = 3;
	SGMatrix<float64_t> orig(data, num_features, num_vectors, false);
	SGMatrix<float64_t> m = orig.clone();
	CDenseFeatures<float64_t>* feats = new CDenseFeatures<float64_t>(m);
	CGaussianKernel* kernel = new CGaussianKernel();
	kernel->set_width(1);
	CKernelPCA kpca(kernel);
	kpca.set_target_dim(2);
	kpca.init(feats);
	SGMatrix<float64_t> embedding = kpca.apply_to_feature_matrix(feats);

	float64_t s;
	// allow embedding with opposite sign
	if ( embedding.matrix[0] > 0)
		s = -1;
	for (index_t i = 0; i < num_features * num_vectors; ++i)
		EXPECT_LE(CMath::abs(embedding.matrix[i] - s * resdata[i]), 1E-6);
}
#endif // HAVE_LAPACK
