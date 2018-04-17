#include <gtest/gtest.h>

#include <shogun/lib/common.h>

#include <cereal/archives/json.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <shogun/classifier/svm/LibSVM.h>
#include <shogun/kernel/GaussianKernel.h>

#include <cstring>
#include <fstream>
#include <shogun/io/SGIO.h>

using namespace shogun;

TEST(Cereal, Json_SGVector_float64_load_equals_saved)
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
	catch (std::exception& e) SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.size(), b.size());
	for (index_t i = 0; i < size; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename.c_str());
}

TEST(Cereal, Json_SGMatrix_float64_load_equals_saved)
{
	const index_t nrows = 2, ncols = 3;
	SGMatrix<float64_t> a(nrows, ncols);
	SGMatrix<float64_t> b;

	for (index_t i = 0; i < nrows * ncols; i++)
		a[i] = i;

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
	catch (std::exception& e) SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.num_rows, b.num_rows);
	EXPECT_EQ(a.num_cols, b.num_cols);
	for (index_t i = 0; i < nrows * ncols; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename.c_str());
}

TEST(Cereal, Json_SGVector_complex128_load_equals_saved)
{
	const index_t size = 5;
	SGVector<complex128_t> a(size);
	SGVector<complex128_t> b(size);

	for (index_t i = 0; i < size; ++i)
		a[i] = complex128_t(i, i * 2);

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
	catch (std::exception& e) SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.size(), b.size());
	for (index_t i = 0; i < size; i++)
	{
		EXPECT_NEAR(i, b[i].real(), 1E-15);
		EXPECT_NEAR(i * 2, b[i].imag(), 1E-15);
	}

	remove(filename.c_str());
}

TEST(Cereal, Json_CerealObject_load_equals_saved)
{
	auto m = some<CLibSVM>();
	auto load_machine = some<CLibSVM>();
	auto k = some<CGaussianKernel>();
	m->put("kernel", k);

	std::string filename = std::tmpnam(nullptr);

	try
	{
		{
			std::ofstream os(filename.c_str());
			cereal::JSONOutputArchive archive(os);
			archive(*m);
		}

		{
			std::ifstream is(filename.c_str());
			cereal::JSONInputArchive archive(is);
			archive(*load_machine);
		}
	}
	catch (std::exception& e) SG_SINFO("Error code: %s \n", e.what());

	EXPECT_TRUE(m->equals(load_machine));

	remove(filename.c_str());
}

#endif // ifndef SWIG
