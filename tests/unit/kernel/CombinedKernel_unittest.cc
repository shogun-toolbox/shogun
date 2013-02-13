#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/io/SerializableAsciiFile.h>
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

TEST(CombinedKernelTest,serialization)
{
	CCombinedKernel* combined = new CCombinedKernel();
	combined->append_kernel(new CGaussianKernel(10, 4));
	combined->append_kernel(new CGaussianKernel(10, 3));
	combined->append_kernel(new CGaussianKernel(10, 9));

	SGVector<float64_t> weights(3);
	weights[0]=17.0;
	weights[1]=23.0;
	weights[2]=42.0;

	combined->set_subkernel_weights(weights);


	CSerializableAsciiFile* outfile = new CSerializableAsciiFile("combined_kernel.weights",'w');
	combined->save_serializable(outfile);
	SG_UNREF(outfile);


	CSerializableAsciiFile* infile = new CSerializableAsciiFile("combined_kernel.weights",'r');
	CCombinedKernel* combined_read = new CCombinedKernel();
	combined_read->load_serializable(infile);
	SG_UNREF(infile);

	CGaussianKernel* k0 = (CGaussianKernel*) combined_read->get_kernel(0);
	CGaussianKernel* k1 = (CGaussianKernel*) combined_read->get_kernel(1);
	CGaussianKernel* k2 = (CGaussianKernel*) combined_read->get_kernel(2);

	EXPECT_EQ(k0->get_width(), 4);
	EXPECT_EQ(k1->get_width(), 3);
	EXPECT_EQ(k2->get_width(), 9);

	SG_UNREF(k0);
	SG_UNREF(k1);
	SG_UNREF(k2);

	SGVector<float64_t> w=combined_read->get_subkernel_weights();
	EXPECT_EQ(weights[0], w[0]);
	EXPECT_EQ(weights[1], w[1]);
	EXPECT_EQ(weights[2], w[2]);
	SG_UNREF(combined);
}
