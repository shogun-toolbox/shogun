#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <gtest/gtest.h>

using namespace shogun;

TEST(SGVectorTest,ctor)
{
	SGVector<float64_t> a(10);
	EXPECT_EQ(a.vlen, 10);

	a.zero();
	for (int i=0; i < 10; ++i)
		EXPECT_EQ(0, a[i]);

	a.set_const(3.3);
	for (int i=0; i < 10; ++i)
		EXPECT_EQ(3.3, a[i]);

	float64_t* a_clone = SGVector<float64_t>::clone_vector(a.vector, a.vlen);
	for (int i=0; i < 10; ++i)
		EXPECT_EQ(a_clone[i], a[i]);

	SGVector<float64_t> b(a_clone, 10);
	EXPECT_EQ(b.vlen, 10);
	for (int i=0; i < 10; ++i)
		EXPECT_EQ(b[i], a[i]);

	/* test copy ctor */
	SGVector<float64_t> c(b);
	EXPECT_EQ(c.vlen, b.vlen);
	for (int i=0; i < c.vlen; ++i)
		EXPECT_EQ(b[i], c[i]);

}

TEST(SGVectorTest,add)
{
	SGVector<float64_t> a(10);
	SGVector<float64_t> b(10);
	a.random(0.0, 1024.0);
	b.random(0.0, 1024.0);
	float64_t* b_clone = SGVector<float64_t>::clone_vector(b.vector, b.vlen);
	SGVector<float64_t> c(b_clone, 10);

	c.add(a);
	for (int i=0; i < c.vlen; ++i)
		EXPECT_EQ(c[i], a[i]+b[i]);

	c = a + a;
	EXPECT_EQ(c.vlen, 10);
	for (int i=0; i < c.vlen; ++i)
		EXPECT_EQ(c[i], 2*a[i]);
}

TEST(SGVectorTest,dot)
{
	SGVector<float64_t> a(10);
	a.random(0.0, 1024.0);
	float64_t dot_val = 0.0;

	for (int32_t i = 0; i < a.vlen; ++i)
		dot_val += a[i]*a[i];

	float64_t error = CMath::abs (dot_val - a.dot(a.vector,a.vector, a.vlen));
	EXPECT_TRUE(error < 10E-10);
}

TEST(SGVectorTest,norm)
{
	SGVector<float64_t> a(10);
	a.random(-50.0, 1024.0);

	/* check l-2 norm */
	float64_t l2_norm = CMath::sqrt(a.dot(a.vector,a.vector, a.vlen));
	float64_t error = CMath::abs(l2_norm - SGVector<float64_t>::twonorm(a.vector, a.vlen));
	EXPECT_TRUE(error < 10E-12);

	float64_t l1_norm = 0.0;
	for (int32_t i = 0; i < a.vlen; ++i)
		l1_norm += CMath::abs(a[i]);
	EXPECT_EQ(l1_norm, SGVector<float64_t>::onenorm(a.vector, a.vlen));

	SGVector<float64_t> b(10);
	b.set_const(1.0);
	EXPECT_EQ(10.0,SGVector<float64_t>::qsq(b.vector, b.vlen, 0.5));

	EXPECT_EQ(100,SGVector<float64_t>::qnorm(b.vector, b.vlen, 0.5));
}

TEST(SGVectorTest,misc)
{
	SGVector<float64_t> a(10);
	a.random(-1024.0, 1024.0);
	
	/* test, min, max, sum */
	float64_t min = 1025, max = -1025, sum = 0.0, sum_abs = 0.0;
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		sum += a[i];
		sum_abs += CMath::abs(a[i]);
		if (a[i] > max)
			max = a[i];
		if (a[i] < min)
			min = a[i];
	}
	
	EXPECT_EQ(min, SGVector<float64_t>::min(a.vector,a.vlen));
	EXPECT_EQ(max, SGVector<float64_t>::max(a.vector,a.vlen));
	EXPECT_EQ(sum, SGVector<float64_t>::sum(a.vector,a.vlen));
	EXPECT_EQ(sum_abs, SGVector<float64_t>::sum_abs(a.vector, a.vlen));

	/* test ::vector_multiply(...) */
	SGVector<float64_t> c(10);
	SGVector<float64_t>::vector_multiply(c.vector, a.vector, a.vector, a.vlen);
	for (int32_t i = 0; i < c.vlen; ++i)
		EXPECT_EQ(c[i], a[i]*a[i]);

	/* test ::add(...) */
	SGVector<float64_t>::add(c.vector, 1.5, a.vector, 1.3, a.vector, a.vlen);
	for (int32_t i = 0; i < a.vlen; ++i)
		EXPECT_EQ(c[i],1.5*a[i]+1.3*a[i]);

	/* tests ::add_scalar */
	SGVector<float64_t>::scale_vector(-1.0,a.vector, a.vlen);
	float64_t* a_clone = SGVector<float64_t>::clone_vector(a.vector, a.vlen);
	SGVector<float64_t> b(a_clone, 10);
	SGVector<float64_t>::add_scalar(1.1, b.vector, b.vlen);
	for (int32_t i = 0; i < b.vlen; ++i)
		EXPECT_EQ(b[i],a[i]+1.1);

	float64_t* b_clone = SGVector<float64_t>::clone_vector(b.vector, b.vlen);
	SGVector<float64_t> d(b_clone, b.vlen);
	SGVector<float64_t>::vec1_plus_scalar_times_vec2(d.vector, 1.3, d.vector, b.vlen);
	for (int32_t i = 0; i < d.vlen; ++i)
		EXPECT_EQ(d[i],b[i]+1.3*b[i]);
}
