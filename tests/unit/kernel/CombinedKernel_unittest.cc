#include <gtest/gtest.h>
#include <shogun/base/ShogunEnv.h>
#include <shogun/features/CombinedFeatures.h>
#include <shogun/features/DenseFeatures.h>
#include <shogun/features/streaming/generators/MeanShiftDataGenerator.h>
#include <shogun/io/fs/FileSystem.h>
#include <shogun/io/serialization/JsonSerializer.h>
#include <shogun/io/serialization/JsonDeserializer.h>
#include <shogun/io/stream/BufferedInputStream.h>
#include <shogun/io/stream/FileInputStream.h>
#include <shogun/io/stream/FileOutputStream.h>
#include <shogun/kernel/CombinedKernel.h>
#include <shogun/kernel/CustomKernel.h>
#include <shogun/kernel/GaussianKernel.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/RandomNamespace.h>

using namespace shogun;

TEST(CombinedKernelTest,test_array_operations)
{
	auto combined = std::make_shared<CombinedKernel>();
	auto gaus_1 = std::make_shared<GaussianKernel>();
	combined->append_kernel(gaus_1);

	auto gaus_2 = std::make_shared<GaussianKernel>();
	combined->append_kernel(gaus_2);

	auto gaus_3 = std::make_shared<GaussianKernel>();
	combined->insert_kernel(gaus_3,1);

	auto gaus_4 = std::make_shared<GaussianKernel>();
	combined->insert_kernel(gaus_4,0);

	EXPECT_EQ(combined->get_num_kernels(),4);

	combined->delete_kernel(2);

	auto k_1 = combined->get_kernel(0);
	EXPECT_EQ(k_1, gaus_4);
	auto k_2 = combined->get_kernel(1);
	EXPECT_EQ(k_2, gaus_1);
	auto k_3 = combined->get_kernel(2);
	EXPECT_EQ(k_3, gaus_2);


}

TEST(CombinedKernelTest, test_subset_mixed)
{
	int n_runs = 10;
	int32_t seed = 17;
	std::mt19937_64 prng(seed);

	auto gen = std::make_shared<MeanShiftDataGenerator>(0, 2);
	auto feats = gen->get_streamed_features(n_runs);

	auto feats_combined = std::make_shared<CombinedFeatures>();

	auto combined = std::make_shared<CombinedKernel>();

	auto gaus_1 = std::make_shared<GaussianKernel>(5);
	auto gaus_2 = std::make_shared<GaussianKernel>(5);

	auto gaus_ck = std::make_shared<GaussianKernel>(5);
	gaus_ck->init(feats, feats);

	auto custom_1 = std::make_shared<CustomKernel>(gaus_ck);
	auto custom_2 = std::make_shared<CustomKernel>(gaus_ck);
	;

	combined->append_kernel(custom_1);
	combined->append_kernel(gaus_1);
	feats_combined->append_feature_obj(feats);

	combined->append_kernel(custom_2);
	combined->append_kernel(gaus_2);
	feats_combined->append_feature_obj(feats);

	SGVector<index_t> inds(10);
	inds.range_fill();

	for (index_t i = 0; i < n_runs; ++i)
	{
		random::shuffle(inds, prng);

		feats_combined->add_subset(inds);
		combined->init(feats_combined, feats_combined);

		auto ground_truth_kernel = combined->get_kernel(1);
		auto custom_kernel_1 = combined->get_kernel(0);
		auto custom_kernel_2 = combined->get_kernel(2);

		SGMatrix<float64_t> gauss_matrix =
		    ground_truth_kernel->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix_1 =
		    custom_kernel_1->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix_2 =
		    custom_kernel_2->get_kernel_matrix();

		for (index_t j = 0; j < n_runs; ++j)
		{
			for (index_t k = 0; k < n_runs; ++k)
			{
				EXPECT_NEAR(gauss_matrix(j, k), custom_matrix_1(j, k), 1e-6);
				EXPECT_NEAR(gauss_matrix(j, k), custom_matrix_1(j, k), 1e-6);
			}
		}

		feats_combined->remove_subset();
	}



}

TEST(CombinedKernelTest, test_subset_combined_only)
{
	int n_runs = 10;
	int32_t seed = 17;
	std::mt19937_64 prng(seed);

	auto gen = std::make_shared<MeanShiftDataGenerator>(0, 2);
	auto feats = gen->get_streamed_features(n_runs);

	auto combined = std::make_shared<CombinedKernel>();

	auto gaus_ck = std::make_shared<GaussianKernel>(5);
	gaus_ck->init(feats, feats);

	auto custom_1 = std::make_shared<CustomKernel>(gaus_ck);
	auto custom_2 = std::make_shared<CustomKernel>(gaus_ck);

	combined->append_kernel(custom_1);
	combined->append_kernel(custom_2);

	SGVector<index_t> inds(n_runs);
	inds.range_fill();

	for (index_t i = 0; i < n_runs; ++i)
	{
		random::shuffle(inds, prng);

		feats->add_subset(inds);
		combined->init(feats, feats);
		gaus_ck->init(feats, feats);

		auto custom_kernel_1 = combined->get_kernel(0);
		auto custom_kernel_2 = combined->get_kernel(1);

		SGMatrix<float64_t> gauss_matrix = gaus_ck->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix_1 =
		    custom_kernel_1->get_kernel_matrix();
		SGMatrix<float64_t> custom_matrix_2 =
		    custom_kernel_2->get_kernel_matrix();

		for (index_t j = 0; j < n_runs; ++j)
		{
			for (index_t k = 0; k < n_runs; ++k)
			{
				EXPECT_NEAR(gauss_matrix(j, k), custom_matrix_1(j, k), 1e-6);
				EXPECT_NEAR(gauss_matrix(j, k), custom_matrix_1(j, k), 1e-6);
			}
		}

		feats->remove_subset();
	}



}

TEST(CombinedKernelTest,weights)
{
	auto combined = std::make_shared<CombinedKernel>();
	combined->append_kernel(std::make_shared<GaussianKernel>());
	combined->append_kernel(std::make_shared<GaussianKernel>());
	combined->append_kernel(std::make_shared<GaussianKernel>());

	SGVector<float64_t> weights(3);
	weights[0]=17.0;
	weights[1]=23.0;
	weights[2]=42.0;

	combined->set_subkernel_weights(weights);

	SGVector<float64_t> w=combined->get_subkernel_weights();

	EXPECT_EQ(weights[0], w[0]);
	EXPECT_EQ(weights[1], w[1]);
	EXPECT_EQ(weights[2], w[2]);

}

//FIXME
TEST(CombinedKernelTest, DISABLED_serialization)
{
	auto combined = std::make_shared<CombinedKernel>();
	combined->append_kernel(std::make_shared<GaussianKernel>(10, 4));
	combined->append_kernel(std::make_shared<GaussianKernel>(10, 3));
	combined->append_kernel(std::make_shared<GaussianKernel>(10, 9));

	SGVector<float64_t> weights(3);
	weights[0]=17.0;
	weights[1]=23.0;
	weights[2]=42.0;

	combined->set_subkernel_weights(weights);

	auto fs = env();
	std::string filename("combined_kernel.weights");
	ASSERT_TRUE(fs->file_exists(filename));
	std::unique_ptr<io::WritableFile> file;
	ASSERT_FALSE(fs->new_writable_file(filename, &file));
	auto fos = std::make_shared<io::FileOutputStream>(file.get());
	auto serializer = std::make_unique<io::JsonSerializer>();
	serializer->attach(fos);
	serializer->write(combined);

	std::unique_ptr<io::RandomAccessFile> raf;
	ASSERT_FALSE(fs->new_random_access_file(filename, &raf));
	auto fis = std::make_shared<io::FileInputStream>(raf.get());
	auto bis = std::make_shared<io::BufferedInputStream>(fis.get());
	auto deserializer = std::make_unique<io::JsonDeserializer>();
	deserializer->attach(bis);
	auto deser_obj = deserializer->read_object();
	auto combined_read = deser_obj->as<CombinedKernel>();
	ASSERT_FALSE(fs->delete_file(filename));

	auto k0 = combined_read->get_kernel(0)->as<GaussianKernel>();
	auto k1 = combined_read->get_kernel(1)->as<GaussianKernel>();
	auto k2 = combined_read->get_kernel(2)->as<GaussianKernel>();

	EXPECT_NEAR(k0->get_width(), 4, 1e-9);
	EXPECT_NEAR(k1->get_width(), 3, 1e-9);
	EXPECT_NEAR(k2->get_width(), 9, 1e-9);

	SGVector<float64_t> w=combined_read->get_subkernel_weights();
	EXPECT_EQ(weights[0], w[0]);
	EXPECT_EQ(weights[1], w[1]);
	EXPECT_EQ(weights[2], w[2]);
}

TEST(CombinedKernelTest,combination)
{
	std::vector<std::vector<std::shared_ptr<Kernel>>> kernel_list;
	auto combined_list = CombinedKernel::combine_kernels(kernel_list);
	EXPECT_EQ(0, combined_list.size());

	combined_list = CombinedKernel::combine_kernels(kernel_list);
	EXPECT_EQ(0, combined_list.size());

	std::vector<std::shared_ptr<Kernel>> sub_list_1;
	auto ck1 = std::make_shared<GaussianKernel>(10,3);
	sub_list_1.push_back(ck1);
	auto ck2 = std::make_shared<GaussianKernel>(10,5);
	sub_list_1.push_back(ck2);
	auto ck3 = std::make_shared<GaussianKernel>(10,7);
	sub_list_1.push_back(ck3);
	kernel_list.push_back(sub_list_1);

	float64_t combs1[3]= {3, 5, 7};

	combined_list = CombinedKernel::combine_kernels(kernel_list);
	index_t i = 0;
	for (const auto& kernel: combined_list)
	{
		auto c_kernel = kernel->as<CombinedKernel>();
		auto subkernel = c_kernel->get_first_kernel();
		auto c_subkernel = subkernel->as<GaussianKernel>();

		EXPECT_EQ(c_kernel->get_num_subkernels(), 1);
		EXPECT_EQ(c_subkernel->get_width(), combs1[i++]);
	}


	std::vector<std::shared_ptr<Kernel>> sub_list_2;
	auto ck4 = std::make_shared<GaussianKernel>(20,21);
	sub_list_2.push_back(ck4);
	auto ck5 = std::make_shared<GaussianKernel>(20,31);
	sub_list_2.push_back(ck5);
	kernel_list.push_back(sub_list_2);

	float64_t combs2[2][6] = {{   3,   5,    7,  3,   5,    7},
											{ 21, 21, 21, 31, 31, 31}};

	combined_list = CombinedKernel::combine_kernels(kernel_list);

	index_t j = 0;
	for (const auto& kernel: combined_list)
	{
		auto c_kernel = kernel->as<CombinedKernel>();
		EXPECT_EQ(c_kernel->get_num_subkernels(), 2);
		i = 0;
		for (index_t k_idx=0; k_idx<c_kernel->get_num_kernels(); k_idx++)
		{
			auto c_subkernel =
					c_kernel->get_kernel(k_idx)->as<GaussianKernel>();
			EXPECT_NEAR(c_subkernel->get_width(), combs2[i++][j], 1e-9);
		}
		++j;
	}

	std::vector<std::shared_ptr<Kernel>> sub_list_3;
	auto ck6 = std::make_shared<GaussianKernel>(25, 109);
	sub_list_3.push_back(ck6);
	auto ck7 = std::make_shared<GaussianKernel>(25, 203);
	sub_list_3.push_back(ck7);
	auto ck8 = std::make_shared<GaussianKernel>(25, 308);
	sub_list_3.push_back(ck8);
	auto ck9 = std::make_shared<GaussianKernel>(25, 404);
	sub_list_3.push_back(ck9);
	kernel_list.push_back(sub_list_3);

	float64_t combs[3][24] = {
		{	3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7,		3,		5,		7},
		{	21,	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31,	21,	21,	21,	31,	31,	31},
		{	109,	109,	109,	109,	109,	109,	203,	203,	203,	203,	203,	203,	308,	308,	308,	308,	308,	308,	404,	404,	404,	404,	404,	404}
		};

	combined_list = CombinedKernel::combine_kernels(kernel_list);

	j = 0;
	for (const auto& kernel: combined_list)
	{
		auto c_kernel = kernel->as<CombinedKernel>();
		i = 0;
		EXPECT_EQ(c_kernel->get_num_subkernels(), 3);
		for (index_t k_idx=0; k_idx<c_kernel->get_num_kernels(); k_idx++)
		{
			auto c_subkernel =
					c_kernel->get_kernel(k_idx)->as<GaussianKernel>();
			EXPECT_NEAR(c_subkernel->get_width(), combs[i++][j], 1e-9);
		}
		++j;
	}
}
