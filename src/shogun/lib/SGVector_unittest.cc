#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

static float64_t a_sample[] = {0,1,2,3,4,5,6,7,8,9};

TEST(SGVectorTest,ctor)
{
	SGVector<float64_t> a(10);
	EXPECT_EQ(a.vlen, 10);

	a.zero();
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(0, a[i]);
	}

	a.set_const(3.3);
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(3.3, a[i]);
	}

	float64_t* a_clone = SGVector<float64_t>::clone_vector(a_sample, 10);
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(a_clone[i], a_sample[i]);
	}

	SGVector<float64_t> b(a_clone, 10);
	EXPECT_EQ(b.vlen, 10);
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(b[i], a_sample[i]);
	}
}

TEST(SGVectorTest,add)
{
	float64_t* a_clone = SGVector<float64_t>::clone_vector(a_sample, 10);
	SGVector<float64_t> a(a_clone, 10);
	SGVector<float64_t> b(10);
	b.zero();

	a.add(b);
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(a[i], a_sample[i]);
	}

	SGVector<float64_t> c = a + a;
	EXPECT_EQ(c.vlen, 10);
	for (int i=0; i < 10; ++i)
	{
		EXPECT_EQ(c[i], 2*a_sample[i]);
	}
}

TEST(SGVectorTest,dot)
{
	float64_t* a_clone = SGVector<float64_t>::clone_vector(a_sample, 10);
	SGVector<float64_t> a(a_clone, 10);

	EXPECT_EQ(285, a.dot(a.vector,a.vector, a.vlen));
}

TEST(SGVectorTest,norm)
{
	EXPECT_EQ(CMath::sqrt(285.0),SGVector<float64_t>::twonorm(a_sample, 10));

	EXPECT_EQ(45,SGVector<float64_t>::onenorm(a_sample, 10));

	SGVector<float64_t> b(10);
	b.set_const(1.0);
	EXPECT_EQ(10.0,SGVector<float64_t>::qsq(b.vector, 10, 0.5));

	EXPECT_EQ(100,SGVector<float64_t>::qnorm(b.vector, 10, 0.5));
}

TEST(SGVectorTest,misc)
{
	EXPECT_EQ(0.0, SGVector<float64_t>::min(a_sample,10));
	EXPECT_EQ(9.0, SGVector<float64_t>::max(a_sample,10));
	EXPECT_EQ(45.0, SGVector<float64_t>::sum(a_sample,10));

	float64_t* a_clone = SGVector<float64_t>::clone_vector(a_sample, 10);
	SGVector<float64_t> a(a_clone, 10);
	SGVector<float64_t> c(10);
	SGVector<float64_t>::vector_multiply(c.vector, a.vector, a.vector, 10);
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		EXPECT_EQ(c[i],a_sample[i]*a_sample[i]);
	}

	SGVector<float64_t>::add(c.vector, 1.5, a.vector, 1.3, a.vector, 10);
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		EXPECT_EQ(c[i],1.5*a_sample[i]+1.3*a_sample[i]);
	}

	SGVector<float64_t>::scale_vector(-1.0,a.vector,a.vlen);
	EXPECT_EQ(45.0, SGVector<float64_t>::sum_abs(a.vector,a.vlen));

	SGVector<float64_t>::scale_vector(-1.0,a.vector,a.vlen);
	SGVector<float64_t>::add_scalar(1.1, a.vector, a.vlen);
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		EXPECT_EQ(a[i],a_sample[i]+1.1);
	}

	float64_t* b_clone = SGVector<float64_t>::clone_vector(a_sample, 10);
	SGVector<float64_t> b(b_clone,10);
	SGVector<float64_t>::vec1_plus_scalar_times_vec2(b.vector, 1.3, b.vector, 10);
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		EXPECT_EQ(b[i],a_sample[i]+1.3*a_sample[i]);
	}
}
