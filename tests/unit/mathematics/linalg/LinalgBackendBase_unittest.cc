#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <gtest/gtest.h>

using namespace shogun;
using namespace linalg;

TEST(LinalgBackendBase, SGVector_to_gpu_without_gpu_backend)
{
	sg_linalg->set_gpu_backend(nullptr);

	const index_t size = 10;
	SGVector<int32_t> a(size), b(size);
	a.range_fill(0);
	b = to_gpu(a);

	EXPECT_FALSE(a.on_gpu());
	EXPECT_FALSE(b.on_gpu());
}
