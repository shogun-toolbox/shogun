#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LinalgRefactor, CPU_dot)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());
	const index_t size = 10;
	SGVector<int32_t> a(size), b(size);
	a.set_const(1);
	b.set_const(2);

	auto result = linalgns::dot(a, b);

	EXPECT_NEAR(result, 20, 1E-15);
}

#ifdef HAVE_VIENNACL
TEST(LinalgRefactor, transfer_between_GPU_and_CPU)
{
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());
	const index_t size = 10;
	SGVector<int32_t> a(size), b;
	a.range_fill(0);

	//SGVector<int32_t> b(linalgns::to_gpu(a));

	b = linalgns::to_gpu(a);
	//c = linalgns::from_gpu(b);

	//for (index_t i = 0; i < size; ++i)
	//	EXPECT_NEAR(a[i], c[i], 1E-15);
}
#endif
