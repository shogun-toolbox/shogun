#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <shogun/clustering/KMeansBase.h>
#include <cereal/archives/json.hpp>
#include <fstream>


using namespace shogun;

TEST(Cereal, Json_SGVector_FLOAT64_save_load_pair)
{
	SGVector<float64_t> a(2), b;
	a.set_const(1.14263158);

	{
		std::ofstream os("cereatltest_sgvector.cereal");
		cereal::JSONOutputArchive archive(os);
		archive(a);
	}

	{
		std::ifstream is("cereatltest_sgvector.cereal");
		cereal::JSONInputArchive archive(is);
		archive(b);
	}

	for (index_t i = 0; i < 2; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);
}

TEST(Cereal, Json_SGObject_save_load_pair)
{
	CKMeansBase obj_save, obj_load;

	{
		std::ofstream os("cereatltest_save_sgobject.cereal");
		cereal::JSONOutputArchive archive(os);
		archive(obj_save);
	}

	{
		std::ifstream is("cereatltest_save_sgobject.cereal");
		cereal::JSONInputArchive archive(is);
		archive(obj_load);
	}

}
