#include <shogun/lib/config.h>
#include <shogun/mathematics/Random.h>
#include <shogun/mathematics/Math.h>
#include <shogun/lib/external/SFMT/SFMT.h>
#include <shogun/lib/external/dSFMT/dSFMT.h>
#include <shogun/mathematics/Statistics.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>
#include <stdio.h>

using namespace shogun;

const uint32_t n_runs=1200000;
const uint32_t array_len=23;

/**
 * NOTE: these unit tests were generated with MEXP=19937
 * with other exponents it is expected to fail!
 */

TEST(Random, uint32_t)
{
	CRandom* prng = new CRandom(12345);
	uint32_t r = prng->random_32();
	SG_UNREF(prng);
	EXPECT_EQ(1811630862U, r);
}

TEST(Random, uint64_t)
{
	CRandom* prng = new CRandom(12345);
	uint64_t r = prng->random_64();
	SG_UNREF(prng);
	EXPECT_EQ(18328733385137801998U, r);
}

TEST(Random, fill_array_uint32)
{
	CRandom* prng = new CRandom(12345);
	uint32_t t = 2228230814U;
	SGVector<uint32_t> rv(2*SFMT_N32+1);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N32]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_uint32_simd)
{
	CRandom* prng = new CRandom(12345);
	uint32_t t = 2228230814U;
	SGVector<uint32_t> rv(2*SFMT_N32);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N32]);
}
#endif

TEST(Random, fill_array_uint64)
{
	CRandom* prng = new CRandom(12345);
	uint64_t t = 9564086722318310046U;
	SGVector<uint64_t> rv(2*SFMT_N64+1);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N64]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_uint64_simd)
{
	CRandom* prng = new CRandom(12345);
	uint64_t t = 9564086722318310046U;
	SGVector<uint64_t> rv(2*SFMT_N64);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N64]);
}
#endif

TEST(Random, fill_array_oc)
{
	CRandom* prng = new CRandom(12345);
	float64_t t = 0.25551924513287405;
	SGVector<float64_t> rv(2*dsfmt_get_min_array_size()+1);
	prng->fill_array_oc(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, rv[dsfmt_get_min_array_size()]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_oc_simd)
{
	CRandom* prng = new CRandom(12345);
	float64_t t = 0.25551924513287405;
	SGVector<float64_t> rv(2*dsfmt_get_min_array_size());
	prng->fill_array_oc(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, rv[dsfmt_get_min_array_size()]);
}
#endif

TEST(Random, normal_distrib)
{
	CRandom* prng = new CRandom(12345);
	float64_t t = 75.567130769021162;
	float64_t r = prng->normal_distrib(100.0, 10.0);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, r);
}

TEST(Random, random_uint64_1_2)
{
	CMath::init_random(17);
	for (int32_t i=0; i<10000; i++)
	{
		uint64_t r=CMath::random((uint64_t) 1, (uint64_t) 2);
		EXPECT_TRUE(r == 1 || r == 2);
	}
}

TEST(Random, random_uint64_0_10)
{
	CMath::init_random(17);
	int rnds[10] = {0,0,0,0,0,0};
	for (int32_t i=0; i<10000; i++)
	{
		uint64_t r=CMath::random((uint64_t) 0, (uint64_t) 9);
		rnds[r]++;
	}

	for (int32_t i=0; i<10; i++) {
		EXPECT_TRUE(rnds[i]>0);
	}
}

TEST(Random, random_int64_1_2)
{
	CMath::init_random(17);
	for (int32_t i=0; i<10000; i++)
	{
		int64_t r=CMath::random((int64_t) 1, (int64_t) 2);
		EXPECT_TRUE(r == 1 || r == 2);
	}
}

TEST(Random, random_int64_0_10)
{
	CMath::init_random(17);
	int rnds[10] = {0,0,0,0,0,0};
	for (int32_t i=0; i<10000; i++)
	{
		int64_t r=CMath::random((int64_t) 0, (int64_t) 9);
		rnds[r]++;
	}

	for (int32_t i=0; i<10; i++) {
		EXPECT_TRUE(rnds[i]>0);
	}
}

TEST(Random, random_uint32_1_2)
{
	CMath::init_random(17);
	for (int32_t i=0; i<10000; i++)
	{
		uint32_t r=CMath::random((uint32_t) 1, (uint32_t) 2);
		EXPECT_TRUE(r == 1 || r == 2);
	}
}

TEST(Random, random_uint32_0_10)
{
	CMath::init_random(17);
	int rnds[10] = {0,0,0,0,0,0};
	for (int32_t i=0; i<10000; i++)
	{
		uint32_t r=CMath::random((uint32_t) 0, (uint32_t) 9);
		rnds[r]++;
	}

	for (int32_t i=0; i<10; i++) {
		EXPECT_TRUE(rnds[i]>0);
	}
}

TEST(Random, random_int32_1_2)
{
	CMath::init_random(17);
	for (int32_t i=0; i<10000; i++)
	{
		int32_t r=CMath::random((int32_t) 1, (int32_t) 2);
		EXPECT_TRUE(r == 1 || r == 2);
	}
}

TEST(Random, random_int64_range)
{
	CMath::init_random(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		int64_t r=CMath::random((int64_t) 0, (int64_t) array_len-1);
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
}

TEST(Random, random_uint64_range)
{
	CMath::init_random(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		uint64_t r=CMath::random((uint64_t) 0, (uint64_t) array_len-1);
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
}

TEST(Random, random_int32_range)
{
	CMath::init_random(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		int32_t r=CMath::random((int32_t) 0, (int32_t) array_len-1);
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
}

TEST(Random, random_uint32_range)
{
	CMath::init_random(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		uint32_t r=CMath::random((uint32_t) 0, (uint32_t) array_len-1);
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
}

TEST(Random, random_uint32_random_range)
{
	CRandom* prng = new CRandom();
	prng->set_seed(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		uint32_t r=prng->random_32() % array_len;
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
	SG_UNREF(prng);
}

TEST(Random, random_float64_range)
{
	CMath::init_random(17);
	int rnds[array_len];
	for (uint32_t i=0; i<array_len; i++)
		rnds[i]=0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		int32_t r= (int32_t) CMath::random((float64_t) 0, (float64_t) array_len);
		rnds[r]++;
	}

	for (uint32_t i=0; i<array_len; i++) {
		double pbin=double(rnds[i])/n_runs*100*array_len;
		EXPECT_GE(pbin, 99.0);
	}
}

TEST(Random, random_float64_range2)
{
	CMath::init_random(12345678);
	float64_t min=1.0;
	float64_t max=0.0;
	for (uint32_t i=0; i<n_runs; i++)
	{
		float64_t r=CMath::random((float64_t) 0, (float64_t) 1.0);
		min=CMath::min(min, r);
		max=CMath::max(max, r);
	}
	EXPECT_GE(max, 0.99999);
	EXPECT_LE(min, 0.00001);
}

TEST(Random, random_std_normal_quantiles)
{
	CRandom* rand=new CRandom();

	int64_t m=10000000;
	SGVector<int64_t> counts(10);
	counts.zero();

	for (int64_t i=0; i<m; ++i)
	{
		float64_t quantile=CStatistics::normal_cdf(rand->std_normal_distrib(), 1);
		index_t idx=(int32_t)(quantile*counts.vlen);
		counts[idx]++;
	}

	SG_UNREF(rand);

	for (index_t i=0; i<counts.vlen; ++i)
		EXPECT_NEAR(counts[i], m/counts.vlen, m/counts.vlen/200);
}
