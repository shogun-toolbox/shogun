/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2013 Viktor Gal
 * Copyright (C) 2013 Viktor Gal
 */

#ifndef __RANDOM_H__
#define __RANDOM_H__

#include <shogun/base/SGObject.h>

#include <shogun/lib/external/SFMT/SFMT.h>
#include <shogun/lib/external/dSFMT/dSFMT.h>

#define RAND_MAX_32 (float32_t)UINT32_MAX
#define RAND_MAX_64 (float64_t)UINT64_MAX

namespace shogun
{
	/** @breif: PRNG */
	class CRandom : public CSGObject
	{
		public:
			/** default ctor */
			CRandom();

			/** ctor
			 * @param seed the seed for the PRNG
			 */
			CRandom(uint32_t seed);

			/** dtor */
			virtual ~CRandom();

			/** set seed
			 *
			 * @param seed seed for PRNG
			 */
			void set_seed(uint32_t seed);

			/** get seed
			 * 
			 * @return seed
			 */
			uint32_t get_seed() const;

			/**
			 * Generate an unsinged 32-bit random integer
			 *
			 * @return the random 32-bit unsigned integer
			 */
			uint32_t random_32() const;

			/**
			 * Generate an unsinged 64-bit random integer
			 *
			 * @return the random 64-bit unsigned integer
			 */			
			uint64_t random_64() const;

			/**
			 * Fill an array of unsinged 32 bit integer
			 *
			 * @param array 32-bit unsigened int array to be filled
			 * @param size size of the array
			 */
			void fill_array(uint32_t* array, int32_t size) const;

			/**
			 * Fill an array of unsinged 64 bit integer
			 *
			 * @param array 64-bit unsigened int array to be filled
			 * @param size size of the array
			 */
			void fill_array(uint64_t* array, int32_t size) const;

			/**
			 * Fills an array of float64_t with randoms 
			 * from the (0,1] interval
			 *
			 * @param array
			 * @param size
			 */
			void fill_array_oc(float64_t* array, int32_t size) const;

			/**
			 * Fills an array of float64_t with randoms 
			 * from the [0,1) interval
			 *
			 * @param array
			 * @param size
			 */
			void fill_array_co(float64_t* array, int32_t size) const;

			/**
			 * Fills an array of float64_t with randoms 
			 * from the (0,1) interval
			 *
			 * @param array
			 * @param size
			 */
			void fill_array_oo(float64_t* array, int32_t size) const;

			/**
			 * Fills an array of float64_t with randoms 
			 * from the [1,2) interval
			 *
			 * @param array
			 * @param size
			 */

			void fill_array_c1o2(float64_t* array, int32_t size) const;

			/**
			 * Get random
			 * @return a float64_t random from [0,1] interval
			 */
			float64_t random_close() const;

			/**
			 * Get random
			 * @return a float64_t random from (0,1) interval
			 */			
			float64_t random_open() const;

			/**
			 * Get random
			 * 
			 * @return a float64_t random from [0,1) interval
			 */			
			float64_t random_half_open() const;

			/**
			 * Sample a normal distrbution.
			 * Using Ziggurat algorithm
			 * @TODO check it's correctness
			 *
			 * @param mu mean
			 * @param sigma variance
			 * @return sample from the desired normal distrib
			 */
			float64_t normal_distrib(float64_t mu, float64_t sigma) const;

			/**
			 * Sample a standard normal distribution, 
			 * i.e. mean = 0, var = 1.0
			 * @TODO check it's correctness!
			 *
			 * @return sample from the std normal distrib
			 */
			float64_t std_normal_distrib() const;


			virtual const char* get_name() const { return "Random"; } 

		private:
			/** initialise the object */
			void init();

			/** reinit PRNG 
			 *
			 * @param seed seed for the PRNG
			 */
			 inline void reinit(uint32_t seed)
			 {
#ifdef HAVE_PTHREAD
			 	PTHREAD_LOCK(&m_state_lock);
#endif
			 	m_seed = seed;
		 		sfmt_init_gen_rand(m_sfmt_32, m_seed);
		 		sfmt_init_gen_rand(m_sfmt_64, m_seed);
		 		dsfmt_init_gen_rand(m_dsfmt, m_seed);
#ifdef HAVE_PTHREAD
			 	PTHREAD_UNLOCK(&m_state_lock);
#endif
			 }

			/**
			 * Sample from the distribution tail (defined as having x >= R).
			 * 
			 * @return
			 */
			float64_t sample_tail() const;

			/**
			 * Gaussian probability density function, denormailised, that is, y = e^-(x^2/2).
			 */
			float64_t GaussianPdfDenorm(float64_t x) const;

			/**
			 * Inverse function of GaussianPdfDenorm(x)
			 */
			float64_t GaussianPdfDenormInv(float64_t y) const;

		private:
			/** seed */
			uint32_t m_seed;

			/** SFMT struct for 32-bit random */
			sfmt_t* m_sfmt_32;

			/** SFMT struct for 64-bit random */
			sfmt_t* m_sfmt_64;

			/** dSFMT struct */
			dsfmt_t* m_dsfmt;

			/** Number of blocks */
			int32_t m_blockCount; //= 128;

			/** Right hand x coord of the base rectangle, thus also the left hand x coord of the tail */
    		float64_t m_R;//= 3.442619855899;

    		/** Area of each rectangle (pre-determined/computed for 128 blocks). */
			float64_t m_A;// = 9.91256303526217e-3;

			/** Scale factor for converting a UInt with range [0,0xffffffff] to a double with range [0,1]. */
			float64_t m_uint32ToU;// = 1.0 / (float64_t)UINT32_MAX;

			/** Area A divided by the height of B0 */
			float64_t m_A_div_y0;

			/** top-right position ox rectangle i */
			float64_t* m_x;
			float64_t* m_y;

			/** The proprtion of each segment that is entirely within the distribution, expressed as uint where 
        	  a value of 0 indicates 0% and uint.MaxValue 100%. Expressing this as an integer allows some floating
        	  points operations to be replaced with integer ones.
        	 */
 			uint32_t* m_xComp;

#ifdef HAVE_PTHREAD
			/** state lock */
			PTHREAD_LOCK_T m_state_lock;
#endif

	};
}

#endif /* __RANDOM_H__ */
