#include <shogun/lib/config.h>
#include <shogun/lib/SGVector.h>
#include <memory>
#include <gtest/gtest.h>
#include <shogun/mathematics/linalg/Vector.h>
#include <shogun/mathematics/linalg/GPUVectorImpl.h>

using namespace shogun;

#ifdef HAVE_CXX11

TEST(LinalgVector, deepcopy_constructor_from_SGVector)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	Vector<int32_t> a_copy(a);
	Vector<int32_t> a_vec(a);

	a.range_fill(1);

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a_copy[i], a_vec[i], 1E-15);
}

TEST(LinalgVector, deepcopy_constructor_from_Vector_with_no_GPU)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	Vector<int32_t> vec_1(a);
	Vector<int32_t> vec_2(vec_1);

	for (index_t i = 0; i < size; ++i)
		vec_1[i]++;

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], vec_2[i], 1E-15);
}

#ifdef HAVE_VIENNACL
TEST(LinalgVector, transfer_to_GPU)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	Vector<int32_t> a_vec(a);
	a_vec.transferToGPU();

	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], (*a_vec.m_gpu_impl)[i], 1E-15);
}

TEST(LinalgVector, transfer_to_CPU)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	Vector<int32_t> a_vec(a);
	a_vec.transferToGPU();
	for (index_t i = 0; i < size; ++i)
		(*a_vec.m_gpu_impl)[i] += 1;
	a_vec.transferToCPU();

	a.range_fill(1);
	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], a_vec[i], 1E-15);
}

TEST(LinalgVector, deepcopy_constructor_from_Vector_with_GPU)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);

	a.range_fill(0);
	Vector<int32_t> vec_1(a);

	vec_1.transferToGPU();
	for (index_t i = 0; i < size; ++i)
		(*vec_1.m_gpu_impl)[i] += 1;

	Vector<int32_t> vec_2(vec_1);

	for (index_t i = 0; i < size; ++i)
		(*vec_1.m_gpu_impl)[i] += 1;

	a.range_fill(1);
	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], vec_2[i], 1E-15);
}

TEST(LinalgVector, return_SGVector_from_Vector_with_GPU)
{
	const index_t size = 10;
	SGVector<int32_t> a(size);
	a.range_fill(0);

	Vector<int32_t> vec(a);

	vec.transferToGPU();
	for (index_t i = 0; i < size; ++i)
		(*vec.m_gpu_impl)[i] += 1;

	SGVector<int32_t> b(size);
	b = SGVector<int32_t>(vec);

	a.range_fill(1);
	for (index_t i = 0; i < size; ++i)
		EXPECT_NEAR(a[i], b[i], 1E-15);
}

#endif //HAVE_VIENNACL

#endif //HAVE_CXX11
