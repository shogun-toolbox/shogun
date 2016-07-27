#include <shogun/lib/common.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <../tests/unit/io/CerealObject.h>
#include <cereal/archives/json.hpp>

#include <fstream>
#include <string>

using namespace shogun;

TEST(Cereal, Json_SGVector_FLOAT64_load_equals_saved)
{
	const index_t size = 5;
	SGVector<float64_t> a(size);
	SGVector<float64_t> b;
	a.range_fill(1.0);

	char* filename = std::tmpnam(nullptr);

	try
	{
		{
			std::ofstream os(filename);
			cereal::JSONOutputArchive archive(os);
			archive(a);
		}

		{
			std::ifstream is(filename);
			cereal::JSONInputArchive archive(is);
			archive(b);
		}
	}
	catch (std::exception& e)
		SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.vlen, b.vlen);
	for (index_t i = 0; i < size; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename);
}

TEST(Cereal, Json_AnyObject_load_equals_saved)
{
	SGVector<float64_t> vec_a(5);
	SGVector<float64_t> vec_b(5);
	vec_a.range_fill(0);
	vec_b.range_fill(1);
	Any a(vec_a);
	Any b(vec_b);

	char* filename = std::tmpnam(nullptr);

	try
	{
		{
			std::ofstream os(filename);
			cereal::JSONOutputArchive archive(os);
			archive(a);
		}

		{
			std::ifstream is(filename);
			cereal::JSONInputArchive archive(is);
			archive(b);
		}
	}
	catch (std::exception& e)
		SG_SINFO("Error code: %s \n", e.what());

	EXPECT_TRUE(a==b);

	remove(filename);
}
