#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <cereal/archives/json.hpp>
#include <fstream>


using namespace shogun;

TEST(Cereal, Json_vector_FLOAT64_save_load_pair)
{
	SGVector<float64_t> a(2), b;
	a.set_const(1.14263158);

	{
		std::ofstream os("cereatltest_save.cereal");
		cereal::JSONOutputArchive archive(os);
		archive(a);
	}

	{
		std::ifstream is("cereatltest_save.cereal");
		cereal::JSONInputArchive archive(is);
		archive(b);
	}

	for (index_t i = 0; i < 2; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);
}
