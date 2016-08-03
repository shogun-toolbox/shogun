#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <cereal/archives/json.hpp>

#include <shogun/io/SGIO.h>
#include <cstring>
#include <fstream>

using namespace shogun;

TEST(Cereal, Json_SGVector_FLOAT64_load_equals_saved)
{
	const index_t size = 5;
	SGVector<float64_t> a(size);
	SGVector<float64_t> b;
	a.range_fill(1.0);

	std::string filename = std::tmpnam(nullptr);

	try
	{
		{
			std::ofstream os(filename.c_str());
			cereal::JSONOutputArchive archive(os);
			archive(a);
		}

		{
			std::ifstream is(filename.c_str());
			cereal::JSONInputArchive archive(is);
			archive(b);
		}
	}
	catch (std::exception& e)
		SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.size(), b.size());
	EXPECT_EQ(a.ref_count(), b.ref_count());
	for (index_t i = 0; i < size; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename.c_str());
}

TEST(Cereal, Json_SGVector_load_equals_saved_refcounting_false)
{
	const index_t size = 5;
	SGVector<float64_t> a(size, false);
	SGVector<float64_t> b;
	a.range_fill(1.0);

	std::string filename = std::tmpnam(nullptr);

	try
	{
		{
			std::ofstream os(filename.c_str());
			cereal::JSONOutputArchive archive(os);
			archive(a);
		}

		{
			std::ifstream is(filename.c_str());
			cereal::JSONInputArchive archive(is);
			archive(b);
		}
	}
	catch (std::exception& e)
		SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.size(), b.size());
	EXPECT_EQ(-1, b.ref_count());
	for (index_t i = 0; i < size; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename.c_str());
}
