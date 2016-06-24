#include <shogun/lib/config.h>

#include <shogun/mathematics/linalg/LinalgNamespace.h>

#ifdef HAVE_VIENNACL
#include <shogun/mathematics/linalg/LinalgBackendViennaCL.h>
#endif

#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(LinalgRefactor, linalg_cpu_backend_dot)
{
	const index_t size = 10;
	SGVector<int32_t> a(size), b(size);
	a.set_const(1);
	b.set_const(2);

	auto result = linalgns::dot(a, b);

	EXPECT_NEAR(result, 20, 1E-15);
}

TEST(LinalgRefactor, linalg_gpu_backend_dot_without_gpu_backend)
{
	const index_t size = 10;
	SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
	a.set_const(1);
	b.set_const(2);

	a_gpu.set(linalgns::to_gpu(a));
	b_gpu.set(linalgns::to_gpu(b));

	auto result = linalgns::dot(a, b);

	EXPECT_NEAR(result, 20, 1E-15);
}

TEST(LinalgRefactor, linalg_gpu_backend_dot_with_gpu_backend)
{
	#ifdef HAVE_VIENNACL
		sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());
	#endif

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size), a_gpu, b_gpu;
	a.set_const(1);
	b.set_const(2);

	a_gpu.set(linalgns::to_gpu(a));
	b_gpu.set(linalgns::to_gpu(b));

	auto result = linalgns::dot(a, b);

	EXPECT_NEAR(result, 20, 1E-15);
}


TEST(LinalgRefactor, gpu_transfer_between_viennacl_and_cpu_backend)
{
#ifdef HAVE_VIENNACL
	sg_linalg->set_gpu_backend(new LinalgBackendViennaCL());
#endif
	const index_t size = 10;
	SGVector<int32_t> a(size), b, c;
	a.range_fill(0);

	b.set(linalgns::to_gpu(a));
	c.set(linalgns::from_gpu(b));

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], c[i], 1E-15);
}
