#include <kernel/ProductKernel.h>
#include <kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ProductKernelTest,test_array_operations)
{
	CProductKernel* product = new CProductKernel();
	CGaussianKernel* gaus_1 = new CGaussianKernel();
	product->append_kernel(gaus_1);

	CGaussianKernel* gaus_2 = new CGaussianKernel();
	product->append_kernel(gaus_2);

	CGaussianKernel* gaus_3 = new CGaussianKernel();
	product->insert_kernel(gaus_3,1);

	CGaussianKernel* gaus_4 = new CGaussianKernel();
	product->insert_kernel(gaus_4,0);

	EXPECT_EQ(product->get_num_subkernels(),4);

	product->delete_kernel(2);

	CKernel* k_1 = product->get_kernel(0);
	EXPECT_EQ(k_1, gaus_4);
	CKernel* k_2 = product->get_kernel(1);
	EXPECT_EQ(k_2, gaus_1);
	CKernel* k_3 = product->get_kernel(2);
	EXPECT_EQ(k_3, gaus_2);

	SG_UNREF(k_1);
	SG_UNREF(k_2);
	SG_UNREF(k_3);
	SG_UNREF(product);
}
