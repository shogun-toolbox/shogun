#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendViennaCL, add)
{
	const float64_t alpha = 0.3;
	const float64_t beta = -1.5;

	SGVector<float64_t> A(9), A_gpu;
	SGVector<float64_t> B(9), B_gpu;

	for (index_t i = 0; i < 9; ++i)
	{
		A[i] = i;
		B[i] = 0.5*i;
	}

	A_gpu = to_gpu(A);
	B_gpu = to_gpu(B);

	auto result = add(A, B, alpha, beta);

	for (index_t i = 0; i < 9; ++i)
		EXPECT_NEAR(alpha*A[i]+beta*B[i], result[i], 1e-15);
}

TEST(LinalgBackendViennaCL, dot)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());

	const index_t size = 3;
	SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
	a.range_fill(0);
	b.range_fill(0);

	a_gpu = to_gpu(a);
	b_gpu = to_gpu(b);

	auto result = dot(a_gpu, b_gpu);

	EXPECT_NEAR(result, 5, 1E-15);
}

#endif // HAVE_VIENNACL
