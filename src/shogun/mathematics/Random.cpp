/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#include <shogun/mathematics/Random.h>
#include <shogun/base/Parameter.h>
#include <shogun/lib/external/SFMT/SFMT.h>
#include <shogun/lib/external/dSFMT/dSFMT.h>
#include <shogun/lib/Time.h>
#include <shogun/lib/Lock.h>

using namespace shogun;

CRandom::CRandom()
 : m_seed((uint32_t)CTime::get_curtime()*100),
 m_sfmt_32(NULL),
 m_sfmt_64(NULL),
 m_dsfmt(NULL)
{
	init();
}

CRandom::CRandom(uint32_t seed)
 : m_seed(seed),
 m_sfmt_32(NULL),
 m_sfmt_64(NULL),
 m_dsfmt(NULL)
{
	init();
}

CRandom::~CRandom()
{
	SG_FREE(m_x);
	SG_FREE(m_y);
	SG_FREE(m_xComp);
	SG_FREE(m_sfmt_32);
	SG_FREE(m_sfmt_64);
	SG_FREE(m_dsfmt);
}

void CRandom::set_seed(uint32_t seed)
{
	reinit(seed);
}

uint32_t CRandom::get_seed() const
{
	return m_seed;
}

void CRandom::init()
{
	/** init ziggurat variables */
	m_blockCount = 128;
	m_R = 3.442619855899;
	m_A = 9.91256303526217e-3;
	m_uint32ToU = 1.0 / (float64_t)std::numeric_limits<uint32_t>::max();

	m_x = SG_MALLOC(float64_t, m_blockCount + 1);
	m_y = SG_MALLOC(float64_t, m_blockCount);
	m_xComp = SG_MALLOC(uint32_t, m_blockCount);

	// Initialise rectangle position data.
	// m_x[i] and m_y[i] describe the top-right position ox Box i.

	// Determine top right position of the base rectangle/box (the rectangle with the Gaussian tale attached).
	// We call this Box 0 or B0 for short.
	// Note. x[0] also describes the right-hand edge of B1. (See diagram).
	m_x[0] = m_R;
	m_y[0] = GaussianPdfDenorm(m_R);

	// The next box (B1) has a right hand X edge the same as B0.
	// Note. B1's height is the box area divided by its width, hence B1 has a smaller height than B0 because
	// B0's total area includes the attached distribution tail.
	m_x[1] = m_R;
	m_y[1] = m_y[0] + (m_A / m_x[1]);

	// Calc positions of all remaining rectangles.
	for(int i=2; i < m_blockCount; i++)
	{
		m_x[i] = GaussianPdfDenormInv(m_y[i-1]);
		m_y[i] = m_y[i-1] + (m_A / m_x[i]);
	}

	// For completeness we define the right-hand edge of a notional box 6 as being zero (a box with no area).
	m_x[m_blockCount] = 0.0;

	// Useful precomputed values.
	m_A_div_y0 = m_A / m_y[0];

	// Special case for base box. m_xComp[0] stores the area of B0 as a proportion of R
	// (recalling that all segments have area A, but that the base segment is the combination of B0 and the distribution tail).
	// Thus -m_xComp[0] is the probability that a sample point is within the box part of the segment.
	m_xComp[0] = (uint32_t)(((m_R * m_y[0]) / m_A) * (float64_t)std::numeric_limits<uint32_t>::max());

	for(int32_t i=1; i < m_blockCount-1; i++)
	{
		m_xComp[i] = (uint32_t)((m_x[i+1] / m_x[i]) * (float64_t)std::numeric_limits<uint32_t>::max());
	}
	m_xComp[m_blockCount-1] = 0;  // Shown for completeness.

	// Sanity check. Test that the top edge of the topmost rectangle is at y=1.0.
	// Note. We expect there to be a tiny drift away from 1.0 due to the inexactness of floating
	// point arithmetic.
	ASSERT(CMath::abs(1.0 - m_y[m_blockCount-1]) < 1e-10);

	/** init SFMT and dSFMT */
	m_sfmt_32 = SG_MALLOC(sfmt_t, 1);
	m_sfmt_64 = SG_MALLOC(sfmt_t, 1);
	m_dsfmt = SG_MALLOC(dsfmt_t, 1);
	reinit(m_seed);
}

uint32_t CRandom::random_32() const
{
	return sfmt_genrand_uint32(m_sfmt_32);
}

uint64_t CRandom::random_64() const
{
	return sfmt_genrand_uint64(m_sfmt_64);
}

void CRandom::fill_array(uint32_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= sfmt_get_min_array_size32(m_sfmt_32)) && (size % 4) == 0)
	{
		sfmt_fill_array32(m_sfmt_32, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = random_32();
}

void CRandom::fill_array(uint64_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= sfmt_get_min_array_size64(m_sfmt_64)) && (size % 2) == 0)
	{
		sfmt_fill_array64(m_sfmt_64, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = random_64();
}

void CRandom::fill_array_oc(float64_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_open_close(m_dsfmt, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = dsfmt_genrand_open_close(m_dsfmt);
}

void CRandom::fill_array_co(float64_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_close_open(m_dsfmt, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = dsfmt_genrand_close_open(m_dsfmt);
}

void CRandom::fill_array_oo(float64_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_open_open(m_dsfmt, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = dsfmt_genrand_open_open(m_dsfmt);
}

void CRandom::fill_array_c1o2(float64_t* array, int32_t size) const
{
#if defined(USE_ALIGNED_MEMORY) || defined(DARWIN)
	if ((size >= dsfmt_get_min_array_size()) && (size % 2) == 0)
	{
		dsfmt_fill_array_close1_open2(m_dsfmt, array, size);
		return;
	}
#endif
	for (int32_t i=0; i < size; i++)
		array[i] = dsfmt_genrand_close1_open2(m_dsfmt);
}

float64_t CRandom::random_close() const
{
	return sfmt_genrand_real1(m_sfmt_32);
}

float64_t CRandom::random_open() const
{
	return dsfmt_genrand_open_open(m_dsfmt);
}

float64_t CRandom::random_half_open() const
{
	return dsfmt_genrand_close_open(m_dsfmt);
}

float64_t CRandom::normal_distrib(float64_t mu, float64_t sigma) const
{
	return mu + (std_normal_distrib() * sigma);
}

float64_t CRandom::std_normal_distrib() const
{
	for (;;)
	{
		// Select box at random.
		uint8_t u = random_32();
		int32_t i = (int32_t)(u & 0x7F);
		float64_t sign = ((u & 0x80) == 0) ? -1.0 : 1.0;

		// Generate uniform random value with range [0,0xffffffff].
		uint32_t u2 = random_32();

		// Special case for the base segment.
		if(0 == i)
		{
			if(u2 < m_xComp[0])
			{
				// Generated x is within R0.
				return u2 * m_uint32ToU * m_A_div_y0 * sign;
			}
			// Generated x is in the tail of the distribution.
			return sample_tail() * sign;
		}

		// All other segments.
		if(u2 < m_xComp[i])
		{   // Generated x is within the rectangle.
			return u2 * m_uint32ToU * m_x[i] * sign;
		}

		// Generated x is outside of the rectangle.
		// Generate a random y coordinate and test if our (x,y) is within the distribution curve.
		// This execution path is relatively slow/expensive (makes a call to Math.Exp()) but relatively rarely executed,
		// although more often than the 'tail' path (above).
		float64_t x = u2 * m_uint32ToU * m_x[i];
		if(m_y[i-1] + ((m_y[i] - m_y[i-1]) * random_half_open()) < GaussianPdfDenorm(x) ) {
			return x * sign;
		}
	}
}

float64_t CRandom::sample_tail() const
{
	float64_t x, y;
	do
	{
	    x = -CMath::log(random_half_open()) / m_R;
	    y = -CMath::log(random_half_open());
	} while(y+y < x*x);
	return m_R + x;
}

float64_t CRandom::GaussianPdfDenorm(float64_t x) const
{
	return CMath::exp(-(x*x / 2.0));
}

float64_t CRandom::GaussianPdfDenormInv(float64_t y) const
{
    // Operates over the y range (0,1], which happens to be the y range of the pdf,
    // with the exception that it does not include y=0, but we would never call with
    // y=0 so it doesn't matter. Remember that a Gaussian effectively has a tail going
    // off into x == infinity, hence asking what is x when y=0 is an invalid question
    // in the context of this class.
    return CMath::sqrt(-2.0 * CMath::log(y));
}

void CRandom::reinit(uint32_t seed)
{
	m_state_lock.lock();
	m_seed = seed;
	sfmt_init_gen_rand(m_sfmt_32, m_seed);
	sfmt_init_gen_rand(m_sfmt_64, m_seed);
	dsfmt_init_gen_rand(m_dsfmt, m_seed);
	m_state_lock.unlock();
}

