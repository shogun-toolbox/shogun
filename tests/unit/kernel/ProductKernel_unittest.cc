#include <shogun/kernel/ProductKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(ProductKernelTest,test_array_operations)
{
	auto product = std::make_shared<ProductKernel>();
	auto gaus_1 = std::make_shared<GaussianKernel>();
	product->append_kernel(gaus_1);

	auto gaus_2 = std::make_shared<GaussianKernel>();
	product->append_kernel(gaus_2);

	auto gaus_3 = std::make_shared<GaussianKernel>();
	product->insert_kernel(gaus_3,1);

	auto gaus_4 = std::make_shared<GaussianKernel>();
	product->insert_kernel(gaus_4,0);

	EXPECT_EQ(product->get_num_subkernels(),4);

	product->delete_kernel(2);

	auto k_1 = product->get_kernel(0);
	EXPECT_EQ(k_1, gaus_4);
	auto k_2 = product->get_kernel(1);
	EXPECT_EQ(k_2, gaus_1);
	auto k_3 = product->get_kernel(2);
	EXPECT_EQ(k_3, gaus_2);
}
