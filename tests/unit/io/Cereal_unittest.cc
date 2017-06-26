#include "CerealObject.h"
#include <gtest/gtest.h>
#include <shogun/lib/common.h>

#include <cereal/archives/json.hpp>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>

#include <cstring>
#include <fstream>
#include <shogun/io/SGIO.h>

using namespace shogun;

#ifndef SWIG // SWIG should skip this part
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
	EXPECT_EQ(a.ref_count(), b.ref_count());
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
	EXPECT_EQ(a.ref_count(), b.ref_count());
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
	catch (std::exception& e) SG_SINFO("Error code: %s \n", e.what());

	EXPECT_EQ(a.size(), b.size());
	EXPECT_EQ(-1, b.ref_count());
	for (index_t i = 0; i < size; i++)
		EXPECT_NEAR(a[i], b[i], 1E-15);

	remove(filename.c_str());
}

TEST(Cereal, Json_CerealObject_load_equals_saved)
{
	SGVector<float64_t> A(5);
	A.range_fill(0);
	SGVector<float64_t> B(5);
	CCerealObject obj_save(A);
	CCerealObject obj_load;

	std::string filename = std::tmpnam(nullptr);

	obj_save.save_json(filename.c_str());

	obj_load.load_json(filename.c_str());
	B = obj_load.get<SGVector<float64_t>>("test_vector");

	EXPECT_EQ(A.size(), B.size());
	for (index_t i = 0; i < 5; ++i)
		EXPECT_NEAR(A[i], B[i], 1e-15);

	remove(filename.c_str());
}

#endif // ifndef SWIG
