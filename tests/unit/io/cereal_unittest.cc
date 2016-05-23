#include <shogun/lib/common.h>
#include <shogun/base/Parameter.h>
#include <gtest/gtest.h>

#include <shogun/lib/SGVector.h>
#include <shogun/clustering/KMeans.h>
#include <cereal/types/vector.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/archives/json.hpp>
#include <fstream>

using namespace shogun;

TEST(Serialization, Json_vector_FLOAT64)
{
	SGVector<float64_t> a(2);
	SGVector<float64_t> b(2);

	a.set_const(1.14263158);
	b.zero();

	std::ofstream os("sgvector.cereal");
	cereal::JSONOutputArchive archive(os);

	//MyRecord myData;
	archive(a);
}

TEST(Serialization, Json_kmeans)
{
	CKMeans kmeans;

	std::ofstream os("kmeans.cereal");
	cereal::JSONOutputArchive archive(os);

	//MyRecord myData;
	archive(kmeans);
}
