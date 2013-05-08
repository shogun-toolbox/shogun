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
	SG_UNREF(combined_read);
	SG_UNREF(combined);
}

TEST(CombinedKernelTest,combination)
{
	CList* kernel_list = 0;
	
	CList* combined_list = CCombinedKernel::combine_kernels(kernel_list);
	EXPECT_EQ(combined_list->get_num_elements(),0);
	SG_UNREF(combined_list);

	kernel_list = new CList(true);
	combined_list = CCombinedKernel::combine_kernels(kernel_list);
	EXPECT_EQ(combined_list->get_num_elements(),0);
	SG_UNREF(combined_list);
	
	CList* sub_list_1 = new CList(true);
	CGaussianKernel* ck1 = new CGaussianKernel(10,3);
	sub_list_1->append_element(ck1);
	CGaussianKernel* ck2 = new CGaussianKernel(10,5);
	sub_list_1->append_element(ck2);
	CGaussianKernel* ck3 = new CGaussianKernel(10,7);
	sub_list_1->append_element(ck3);
	kernel_list->insert_element(sub_list_1);

	float64_t combs1[3]= {3, 5, 7};
	
	combined_list = CCombinedKernel::combine_kernels(kernel_list);
	index_t i = 0;
	for (CSGObject* kernel=combined_list->get_first_element(); kernel; kernel=combined_list->get_next_element())
	{
		CCombinedKernel* c_kernel = dynamic_cast<CCombinedKernel* >(kernel);
		CSGObject* subkernel = c_kernel->get_first_kernel();
		CGaussianKernel* c_subkernel = dynamic_cast<CGaussianKernel* >(subkernel);

		EXPECT_EQ(c_kernel->get_num_subkernels(), 1);
		EXPECT_EQ(c_subkernel->get_width(), combs1[i++]);

		SG_UNREF(subkernel);
		SG_UNREF(kernel);
	}
	SG_UNREF(combined_list);

	CList * sub_list_2 = new CList(true);
	CGaussianKernel* ck4 = new CGaussianKernel(20,21);
	sub_list_2->append_element(ck4);
	CGaussianKernel* ck5 = new CGaussianKernel(20,31);
	sub_list_2->append_element(ck5);
	kernel_list->append_element(sub_list_2);

	float64_t combs2[2][6] = {{ 21, 21, 21, 31, 31, 31},
											{   3,   5,    7,  3,   5,    7}};

	combined_list = CCombinedKernel::combine_kernels(kernel_list);
	
	index_t j = 0;
	for (CSGObject* kernel=combined_list->get_first_element(); kernel; kernel=combined_list->get_next_element())
	{
		CCombinedKernel* c_kernel = dynamic_cast<CCombinedKernel* >(kernel);
		EXPECT_EQ(c_kernel->get_num_subkernels(), 2);
		i = 0;
		for (CSGObject* subkernel=c_kernel->get_first_kernel(); subkernel; subkernel=c_kernel->get_next_kernel())
		{
			CGaussianKernel* c_subkernel = dynamic_cast<CGaussianKernel* >(subkernel);
			EXPECT_EQ(c_subkernel->get_width(), combs2[i++][j]);
			SG_UNREF(subkernel);
		}
		++j;
		SG_UNREF(kernel);
	}

	SG_UNREF(combined_list);

	CList* sub_list_3 = new CList(true);
	CGaussianKernel* ck6 = new CGaussianKernel(25, 109);
	sub_list_3->append_element(ck6);
	CGaussianKernel* ck7 = new CGaussianKernel(25, 203);
	sub_list_3->append_element(ck7);
	CGaussianKernel* ck8 = new CGaussianKernel(25, 308);
	sub_list_3->append_element(ck8);
	CGaussianKernel* ck9 = new CGaussianKernel(25, 404);
	sub_list_3->append_element(ck9);
	kernel_list->append_element(sub_list_3);

	float64_t combs[3][24] = {
		{	109,	109,	109,	109,	109,	109,	203,	203,	203,	203,	203,	203,	308,	308,	308,	308,	308,	308,	404,	404,	404,	404,	404,	404},
		{	21, 	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31},
		{	3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7}};
	
	combined_list = CCombinedKernel::combine_kernels(kernel_list);
	
	j = 0;
	for (CSGObject* kernel=combined_list->get_first_element(); kernel; kernel=combined_list->get_next_element())
	{
		CCombinedKernel* c_kernel = dynamic_cast<CCombinedKernel* >(kernel);
		i = 0;
		EXPECT_EQ(c_kernel->get_num_subkernels(), 3);
		for (CSGObject* subkernel=c_kernel->get_first_kernel(); subkernel; subkernel=c_kernel->get_next_kernel())
		{
			CGaussianKernel* c_subkernel = dynamic_cast<CGaussianKernel* >(subkernel);
			EXPECT_EQ(c_subkernel->get_width(), combs[i++][j]);
			SG_UNREF(subkernel);
		}
		++j;
		SG_UNREF(kernel);
	}

	SG_UNREF(combined_list);
	SG_UNREF(kernel_list);
}
