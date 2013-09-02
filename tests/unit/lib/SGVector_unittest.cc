#include <shogun/lib/SGVector.h>
#include <shogun/lib/SGMatrix.h>
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

TEST(SGVectorTest,setget)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	EXPECT_EQ(v[0], 12);
	EXPECT_EQ(v[1], 1);
	EXPECT_EQ(v[2], 7);
	EXPECT_EQ(v[3], 9);

	v.set_element(3,0);
	v.set_element(2,1);
	v.set_element(1,2);
	v.set_element(0,3);

	EXPECT_EQ(v[0], 3);
	EXPECT_EQ(v[1], 2);
	EXPECT_EQ(v[2], 1);
	EXPECT_EQ(v[3], 0);
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
	int arg_max = 0, arg_max_abs = 0;
	float64_t min = 1025, max = -1025, sum = 0.0, max_abs = -1, sum_abs = 0.0;
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		sum += a[i];
		sum_abs += CMath::abs(a[i]);
		if (CMath::abs(a[i]) > max_abs)
		{
			max_abs = CMath::abs(a[i]);
			arg_max_abs=i;
		}
		if (a[i] > max)
		{
			max = a[i];
			arg_max=i;
		}
		if (a[i] < min)
			min = a[i];
	}

	EXPECT_EQ(min, SGVector<float64_t>::min(a.vector,a.vlen));
	EXPECT_EQ(max, SGVector<float64_t>::max(a.vector,a.vlen));
	EXPECT_EQ(arg_max, SGVector<float64_t>::arg_max(a.vector,1, a.vlen, NULL));
	EXPECT_EQ(max_abs, SGVector<float64_t>::max_abs(a.vector,a.vlen));
	EXPECT_EQ(arg_max_abs, SGVector<float64_t>::arg_max_abs(a.vector, 1, a.vlen, NULL));
	EXPECT_EQ(sum, SGVector<float64_t>::sum(a.vector,a.vlen));
	EXPECT_DOUBLE_EQ(sum_abs, SGVector<float64_t>::sum_abs(a.vector, a.vlen));

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
		EXPECT_DOUBLE_EQ(d[i],b[i]+1.3*b[i]);
}

TEST(SGVectorTest,complex64_tests)
{
	SGVector<complex64_t> a(10);
	a.set_const(complex64_t(5.0, 6.0));
	SGVector<complex64_t> b=a.clone();

	// test ::operator+ and []
	a=a+b;
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}

	// test ::misc
	SGVector<complex64_t>::vec1_plus_scalar_times_vec2(a.vector,
		complex64_t(0.0, 0.0), b.vector, a.vlen);
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}
	a.permute();
	complex64_t sum=SGVector<complex64_t>::sum_abs(a.vector, 1);
	EXPECT_NEAR(sum.real(), 15.62049935181330878825, 1E-14);
	EXPECT_NEAR(sum.imag(), 0.0, 1E-14);

	SGVector<index_t> res=a.find(complex64_t(10.0, 12.0));
	for (index_t i=0; i<res.vlen; ++i)
		EXPECT_EQ(res[i], i);

	a.scale(complex64_t(1.0));
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}

	// tests ::norm
	float64_t norm1=SGVector<complex64_t>::onenorm(a.vector, 1);
	EXPECT_NEAR(norm1, 15.62049935181330795331, 1E-14);

	complex64_t norm2=SGVector<complex64_t>::twonorm(a.vector, 1);
	EXPECT_NEAR(norm2.real(), 10.0, 1E-14);
	EXPECT_NEAR(norm2.imag(), 12.0, 1E-14);

	// tests ::maths
	a.set_const(complex64_t(1.0, 2.0));
	a.abs();
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 2.23606797749978980505, 1E-14);
		EXPECT_NEAR(a[i].imag(), 0.0, 1E-14);
	}

	a.set_const(complex64_t(1.0, 2.0));
	a.sin();
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 3.16577851321616821068, 1E-14);
		EXPECT_NEAR(a[i].imag(), 1.95960104142160607132, 1E-14);
	}

	a.set_const(complex64_t(1.0, 2.0));
	a.cos();
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 2.03272300701966557313, 1E-14);
		EXPECT_NEAR(a[i].imag(), -3.05189779915179970615, 1E-14);
	}

	a.set_const(complex64_t(1.0, 2.0));
	SGVector<float64_t> a_real=a.get_real();
	SGVector<float64_t> a_imag=a.get_imag();
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a_real[i], 1.0, 1E-14);
		EXPECT_NEAR(a_imag[i], 2.0, 1E-14);
	}
}

TEST(SGVectorTest,equals_equal)
{
	SGVector<float64_t> a(3);
	SGVector<float64_t> b(3);
	a[0]=0;
	a[1]=1;
	a[2]=2;
	b[0]=0;
	b[1]=1;
	b[2]=2;

	EXPECT_TRUE(a.equals(b));
}

TEST(SGVectorTest,equals_different)
{
	SGVector<float64_t> a(3);
	SGVector<float64_t> b(3);
	a[0]=0;
	a[1]=1;
	a[2]=2;
	b[0]=0;
	b[1]=1;
	b[2]=3;

	EXPECT_FALSE(a.equals(b));
}

TEST(SGVectorTest,equals_different_size)
{
	SGVector<float64_t> a(3);
	SGVector<float64_t> b(2);
	a.zero();
	b.zero();

	EXPECT_FALSE(a.equals(b));
}

TEST(SGVectorTest, convert_to_matrix)
{
	index_t len=6;
	index_t nrows=2;
	index_t ncols=3;
	int32_t c_order_memory[]={1, 2, 3, 4, 5, 6};
	int32_t fortran_order_memory[]={1, 4, 2, 5, 3, 6};

	SGVector<int32_t> vector;
	SGMatrix<int32_t> a;
	SGMatrix<int32_t> b;

	vector=SGVector<int32_t>(c_order_memory, len, false);
	a=SGVector<int32_t>::convert_to_matrix(vector, nrows, ncols, false);

	vector=SGVector<int32_t>(fortran_order_memory, len, false);
	b=SGVector<int32_t>::convert_to_matrix(vector, nrows, ncols, true);

	for (index_t i=0; i<nrows; i++)
	{
		for (index_t j=0; j<ncols; j++)
		{
			EXPECT_EQ(a(i, j), b(i, j));
		}
	}
}

TEST(SGVectorTest,qsort)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	v.qsort();

	EXPECT_EQ(v.vlen, 4);
	EXPECT_EQ(v[0], 1);
	EXPECT_EQ(v[1], 7);
	EXPECT_EQ(v[2], 9);
	EXPECT_EQ(v[3], 12);
}

TEST(SGVectorTest,is_sorted)
{
	SGVector<index_t> v(4);
	v[0]=12;
	v[1]=1;
	v[2]=7;
	v[3]=9;

	EXPECT_EQ(v.is_sorted(), false);
	v.qsort();

	EXPECT_EQ(v.is_sorted(), true);
}

TEST(SGVectorTest,is_sorted_0)
{
	SGVector<index_t> v(0);

	EXPECT_EQ(v.is_sorted(), true);
	v.qsort();

	EXPECT_EQ(v.is_sorted(), true);
}

TEST(SGVectorTest,is_sorted_1)
{
	SGVector<index_t> v(1);
	v[0]=12;

	EXPECT_EQ(v.is_sorted(), true);
	v.qsort();

	EXPECT_EQ(v.is_sorted(), true);
}

TEST(SGVectorTest,is_sorted_2)
{
	SGVector<index_t> v(2);
	v[0]=12;
	v[1]=1;

	EXPECT_EQ(v.is_sorted(), false);
	v.qsort();

	EXPECT_EQ(v.is_sorted(), true);
}
