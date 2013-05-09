#include <shogun/lib/config.h>
#include <shogun/mathematics/Random.h>
#include <shogun/lib/SGVector.h>
#include <gtest/gtest.h>

using namespace shogun;

/**
 * NOTE: these unit tests were generated with MEXP=19937
 * with other exponents it is expected to fail!
 */

TEST(Random, uint32_t)
{
	CRandom* prng = new CRandom();
	uint32_t r = prng->random_32();
	SG_UNREF(prng);
	EXPECT_EQ(1811630862U, r);
}

TEST(Random, uint64_t)
{
	CRandom* prng = new CRandom();
	uint64_t r = prng->random_64();
	SG_UNREF(prng);
	EXPECT_EQ(18328733385137801998U, r);
}

TEST(Random, fill_array_uint32)
{
	CRandom* prng = new CRandom();
	uint32_t t = 2228230814U;
	SGVector<uint32_t> rv(2*SFMT_N32+1);
	prng->fill_array(rv.vector, rv.vlen);
	
	EXPECT_EQ(t, rv[SFMT_N32]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_uint32_simd)
{
	CRandom* prng = new CRandom();
	uint32_t t = 2228230814U;
	SGVector<uint32_t> rv(2*SFMT_N32);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N32]);
}
#endif

TEST(Random, fill_array_uint64)
{
	CRandom* prng = new CRandom();
	uint64_t t = 9564086722318310046U;
	SGVector<uint64_t> rv(2*SFMT_N64+1);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N64]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_uint64_simd)
{
	CRandom* prng = new CRandom();
	uint64_t t = 9564086722318310046U;
	SGVector<uint64_t> rv(2*SFMT_N64);
	prng->fill_array(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_EQ(t, rv[SFMT_N64]);
}
#endif

TEST(Random, fill_array_oc)
{
	CRandom* prng = new CRandom();
	float64_t t = 0.25551924513287405;
	SGVector<float64_t> rv(2*dsfmt_get_min_array_size()+1);
	prng->fill_array_oc(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, rv[dsfmt_get_min_array_size()]);
}

#ifdef HAVE_SSE2
TEST(Random, fill_array_oc_simd)
{
	CRandom* prng = new CRandom();
	float64_t t = 0.25551924513287405;
	SGVector<float64_t> rv(2*dsfmt_get_min_array_size());
	prng->fill_array_oc(rv.vector, rv.vlen);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, rv[dsfmt_get_min_array_size()]);
}
#endif

TEST(Random, normal_distrib)
{
	CRandom* prng = new CRandom();
	float64_t t = 75.567130769021162;
	float64_t r = prng->normal_distrib(100.0, 10.0);
	SG_UNREF(prng);

	EXPECT_DOUBLE_EQ(t, r);
}
