#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(CombinedKernelTest,weights)
{
	CCombinedKernel* combined = new CCombinedKernel();
	combined->append_kernel(new CGaussianKernel());
	combined->append_kernel(new CGaussianKernel());
	combined->append_kernel(new CGaussianKernel());

	SGVector<float64_t> weights(3);
	weights[0]=17.0;
	weights[1]=23.0;
	weights[2]=42.0;

	combined->set_subkernel_weights(weights);

	SGVector<float64_t> w=combined->get_subkernel_weights();

	EXPECT_EQ(weights[0], w[0]);
	EXPECT_EQ(weights[1], w[1]);
	EXPECT_EQ(weights[2], w[2]);
	SG_UNREF(combined);
}

