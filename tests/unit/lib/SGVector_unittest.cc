#include <gtest/gtest.h>
#include <shogun/base/range.h>
#include <shogun/lib/SGMatrix.h>
#include <shogun/lib/SGVector.h>
#include <shogun/mathematics/Math.h>
#include <shogun/mathematics/linalg/LinalgNamespace.h>
#include <shogun/mathematics/RandomNamespace.h>
#include <shogun/mathematics/NormalDistribution.h>
#include <shogun/mathematics/eigen3.h>

#include <random>

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

	/* test iterator */
	std::vector<float64_t> src {1.0, 2.0, 3.0, 4.0, 5.0};
	SGVector<float64_t> d(src.begin(), src.end());
	EXPECT_EQ(src.size(), static_cast<size_t>(d.vlen));
	for (int i=0; i < c.vlen; ++i)
		EXPECT_EQ(b[i], c[i]);

	/* test initializer list */
	SGVector<float64_t> e {1.0, 2.0, 3.0, 4.0, 5.0};
	EXPECT_EQ(5, e.vlen);
	EXPECT_EQ(1.0, e[0]);
	EXPECT_EQ(2.0, e[1]);
	EXPECT_EQ(3.0, e[2]);
	EXPECT_EQ(4.0, e[3]);
	EXPECT_EQ(5.0, e[4]);
}

TEST(SGVectorTest, ctor_from_matrix)
{
	const int32_t seed = 100;
	const index_t n_rows = 5, n_cols = 4;

	SGMatrix<float64_t> mat(n_rows, n_cols);
	std::mt19937_64 prng(seed);
	NormalDistribution<float64_t> normal_dist;
	for (index_t i = 0; i < mat.size(); ++i)
		mat[i] = normal_dist(prng);

	auto vec = SGVector<float64_t>(mat);

	for (index_t i = 0; i < n_rows * n_cols; ++i)
		EXPECT_EQ(mat[i], vec[i]);
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
	const int32_t seed = 17;

	SGVector<float64_t> a(10);
	SGVector<float64_t> b(10);

	std::mt19937_64 prng(seed);
	random::fill_array(a, 0.0, 1024.0, prng);
	random::fill_array(b, 0.0, 1024.0, prng);
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

TEST(SGVectorTest,norm)
{
	std::mt19937_64 prng(17);
	SGVector<float64_t> a(10);
	random::fill_array(a, -50.0, 1024.0, prng);

	/* check l-2 norm */
	float64_t l2_norm = std::sqrt(linalg::dot(a, a));
	float64_t sgl2_norm = SGVector<float64_t>::twonorm(a.vector, a.vlen);

	EXPECT_NEAR(l2_norm, sgl2_norm, 1e-12);

	float64_t l1_norm = 0.0;
	for (int32_t i = 0; i < a.vlen; ++i)
		l1_norm += Math::abs(a[i]);
	EXPECT_EQ(l1_norm, SGVector<float64_t>::onenorm(a.vector, a.vlen));

	SGVector<float64_t> b(10);
	b.set_const(1.0);
	EXPECT_EQ(10.0,SGVector<float64_t>::qsq(b.vector, b.vlen, 0.5));

	EXPECT_EQ(100,SGVector<float64_t>::qnorm(b.vector, b.vlen, 0.5));
}

TEST(SGVectorTest,misc)
{
	std::mt19937_64 prng(17);
	SGVector<float64_t> a(10);
	random::fill_array(a, -1024.0, 1024.0, prng);

	/* test, sum */
	float64_t sum = 0.0, sum_abs = 0.0;
	for (int32_t i = 0; i < a.vlen; ++i)
	{
		sum += a[i];
		sum_abs += Math::abs(a[i]);
	}

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

TEST(SGVectorTest,complex128_tests)
{
	SGVector<complex128_t> a(10);
	a.set_const(complex128_t(5.0, 6.0));
	SGVector<complex128_t> b=a.clone();

	// test ::operator+ and []
	a=a+b;
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}

	// test ::misc
	SGVector<complex128_t>::vec1_plus_scalar_times_vec2(a.vector,
		complex128_t(0.0, 0.0), b.vector, a.vlen);
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}

	complex128_t sum=SGVector<complex128_t>::sum_abs(a.vector, 1);
	EXPECT_NEAR(sum.real(), 15.62049935181330878825, 1E-14);
	EXPECT_NEAR(sum.imag(), 0.0, 1E-14);

	SGVector<index_t> res=a.find(complex128_t(10.0, 12.0));
	for (index_t i=0; i<res.vlen; ++i)
		EXPECT_EQ(res[i], i);

	a.scale(complex128_t(1.0));
	for (index_t i=0; i<a.vlen; ++i)
	{
		EXPECT_NEAR(a[i].real(), 10.0, 1E-14);
		EXPECT_NEAR(a[i].imag(), 12.0, 1E-14);
	}

	// tests ::norm
	float64_t norm1=SGVector<complex128_t>::onenorm(a.vector, 1);
	EXPECT_NEAR(norm1, 15.62049935181330795331, 1E-14);

	complex128_t norm2=SGVector<complex128_t>::twonorm(a.vector, 1);
	EXPECT_NEAR(norm2.real(), 10.0, 1E-14);
	EXPECT_NEAR(norm2.imag(), 12.0, 1E-14);

	// tests ::get_real and ::get_imag
	a.set_const(complex128_t(1.0, 2.0));
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

TEST(SGVectorTest, clone_empty)
{
	SGVector<float32_t> vec;
	ASSERT_EQ(vec.data(), nullptr);

	SGVector<float32_t> copy = vec.clone();
	EXPECT_EQ(copy.data(), vec.data());
	EXPECT_TRUE(vec.equals(copy));
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

TEST(SGVectorTest, to_eigen3_column_vector)
{
	const int n = 9;

	SGVector<float64_t> sg_vec(9);
	for (int32_t i=0; i<n; i++)
		sg_vec[i] = i;

	Eigen::Map<Eigen::VectorXd> eigen_vec = sg_vec;

	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(sg_vec[i], eigen_vec[i]);
}

TEST(SGVectorTest, from_eigen3_column_vector)
{
	const int n = 9;

	Eigen::VectorXd eigen_vec(9);
	for (int32_t i=0; i<n; i++)
		eigen_vec[i] = i;

	SGVector<float64_t> sg_vec = eigen_vec;

	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(eigen_vec[i], sg_vec[i]);
}

TEST(SGVectorTest, to_eigen3_row_vector)
{
	const int n = 9;

	SGVector<float64_t> sg_vec(9);
	for (int32_t i=0; i<n; i++)
		sg_vec[i] = i;

	Eigen::Map<Eigen::RowVectorXd> eigen_vec = sg_vec;

	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(sg_vec[i], eigen_vec[i]);
}

TEST(SGVectorTest, from_eigen3_row_vector)
{
	const int n = 9;

	Eigen::RowVectorXd eigen_vec(9);
	for (int32_t i=0; i<n; i++)
		eigen_vec[i] = i;

	SGVector<float64_t> sg_vec = eigen_vec;

	for (int32_t i=0; i<n; i++)
		EXPECT_EQ(eigen_vec[i], sg_vec[i]);
}

TEST(SGVectorTest, resize_vector_larger)
{
	index_t len = 3;
	SGVector<index_t> m(len);
	SGVector<index_t> m_ref(len);

	for (index_t i=0; i<len; i++)
	{
		m[i]=i;
		m_ref[i]=i;
	}

	index_t new_len = 5;
	m.resize_vector(new_len);
	EXPECT_EQ(m.vlen, new_len);

	// check for "old" block to be intact
	for (index_t i=0; i<len; i++)
		EXPECT_EQ(m[i], m_ref[i]);

	// check zero padding of added elements
	for (index_t i=len; i<new_len; i++)
		EXPECT_EQ(m[i], 0);
}

TEST(SGVectorTest,iterator)
{
	SGVector<float64_t> t {1.0, 2.0, 3.0, 4.0, 5.0};

	auto begin = t.begin();
	auto end = t.end();
	EXPECT_EQ(t.vlen, std::distance(begin, end));
	EXPECT_EQ(1.0, *begin++);
	++begin;
	EXPECT_EQ(3.0, *begin);
	--begin;
	EXPECT_EQ(2.0, *begin);
	EXPECT_TRUE(begin != end);
	begin += 2;
	EXPECT_EQ(4.0, *begin);
	begin -= 1;
	EXPECT_EQ(3.0, *begin);
	EXPECT_EQ(4.0, begin[1]);
	auto new_begin = begin + 2;
	EXPECT_EQ(5.0, *new_begin);
	EXPECT_TRUE(++new_begin == end);

	// range-based loop should work as well
	auto index = 0;
	for (auto v: t)
		EXPECT_EQ(t[index++], v);
}

TEST(SGVectorTest,unique)
{
    SGVector<int32_t> vec{1,2,3,1,2,3,3,4,5,4,5,6,7};
    auto num_unique = vec.unique(vec.vector, vec.vlen);

    EXPECT_EQ(7, num_unique);
    for (index_t i = 0; i < num_unique; ++i)
		EXPECT_EQ(i+1, vec[i]);
}

TEST(SGVectorTest, unique_method)
{
	SGVector<int32_t> vec{1, 4, 3, 1, 4, 3, 3};
	auto vec_unique = vec.unique();

	ASSERT_NE(vec_unique.data(), nullptr);
	ASSERT_EQ(3, vec_unique.size());
	EXPECT_NE(vec_unique.data(), vec.data());
	EXPECT_EQ(vec_unique[0], 1);
	EXPECT_EQ(vec_unique[1], 3);
	EXPECT_EQ(vec_unique[2], 4);
}

TEST(SGVectorTest, as)
{
	std::vector<float64_t> data{4, 3, 2, 1};
	SGVector<float64_t> vec{data.begin(), data.end()};

	SGVector<float32_t> vec_float = vec.as<float32_t>();
	ASSERT_NE(vec_float.data(), nullptr);
	EXPECT_NE((void*)vec_float.data(), (void*)vec.data());
	ASSERT_EQ(vec_float.size(), vec.size());
	for (auto i : range(vec.size()))
	{
		EXPECT_NEAR((float32_t)vec[i], vec_float[i], 1e-7);
		EXPECT_NEAR((float32_t)data[i], vec_float[i], 1e-7);
	}

	SGVector<int32_t> vec_int = vec.as<int32_t>();
	ASSERT_NE(vec_int.data(), nullptr);
	EXPECT_NE((void*)vec_int.data(), (void*)vec.data());
	ASSERT_EQ(vec_int.size(), vec.size());
	for (auto i : range(vec.size()))
	{
		EXPECT_EQ((int32_t)vec[i], vec_int[i]);
		EXPECT_EQ((int32_t)data[i], vec_int[i]);
	}
}

TEST(SGVectorTest, slice)
{
	index_t vlen = 100;
	index_t l = 20;
	index_t h = 50;
	index_t l2 = 10;
	index_t h2 = 20;
	index_t c1 = 0;
	index_t c2 = 1;

	SGVector<float64_t> vec(vlen);
	vec.range_fill();

	auto sliced_vec = vec.slice(l, h);
	EXPECT_EQ(sliced_vec.size(), h - l);
	for (index_t i = 0; i < h - l; i++)
	{
		EXPECT_EQ(vec[i + l], sliced_vec[i]);
	}

	auto sliced_vec2 = sliced_vec.slice(l2, h2);
	EXPECT_EQ(sliced_vec2.size(), h2 - l2);
	for (index_t i = 0; i < h2 - l2; i++)
	{
		EXPECT_EQ(sliced_vec[i + l2], sliced_vec2[i]);
	}

	sliced_vec.set_const(c1);
	auto sliced_vec3 = sliced_vec2.slice(0, sliced_vec2.vlen);
	EXPECT_EQ(sliced_vec2, sliced_vec3);
	sliced_vec3.set_const(c2);

	for (index_t i = 0; i < vlen; i++)
	{
		if (i < l || i >= h)
			EXPECT_EQ(vec[i], i);
		else if (i < l + l2 || i >= l + h2)
			EXPECT_EQ(vec[i], sliced_vec[i - l]);
		else
			EXPECT_EQ(vec[i], sliced_vec2[i - l - l2]);
	}
}
