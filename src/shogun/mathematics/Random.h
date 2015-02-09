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

#include <shogun/lib/config.h>

#include <shogun/base/SGObject.h>
#include <shogun/lib/config.h>
#include <shogun/lib/Lock.h>
#include <limits>

/* opaque pointers */
struct SFMT_T;
struct DSFMT_T;

namespace shogun
{
	class CLock;
	/** @brief: Pseudo random number geneartor
	 *
	 * It is based on SIMD oriented Fast Mersenne Twister(SFMT) pseudorandom
	 * number generator.
	 *
	 * */
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
			 * Generate an unsigned 32-bit random integer
			 *
			 * @return the random 32-bit unsigned integer
			 */
			uint32_t random_32() const;

			/**
			 * Generate an unsigned 64-bit random integer
			 *
			 * @return the random 64-bit unsigned integer
			 */
			uint64_t random_64() const;

			/**
			 * Generate a signed 32-bit random integer
			 *
			 * @return the random 32-bit signed integer
			 */
			inline int32_t random_s32() const
			{
				return random_32() & ((uint32_t(-1)<<1)>>1);
			}

			/**
			 * Generate a signed 64-bit random integer
			 *
			 * @return the random 64-bit signed integer
			 */
			inline int64_t random_s64() const
			{
				return random_64() & ((uint64_t(-1)<<1)>>1);
			}

			/** generate an unsigned 64bit integer in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline uint64_t random(uint64_t min_value, uint64_t max_value)
			{
				return min_value + random_64() % (max_value-min_value+1);
			}

			/** generate an signed 64bit integer in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline int64_t random(int64_t min_value, int64_t max_value)
			{
				return min_value + random_s64() % (max_value-min_value+1);
			}

			/** generate an unsigned signed 32bit integer in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline uint32_t random(uint32_t min_value, uint32_t max_value)
			{
				return min_value + random_32() % (max_value-min_value+1);
			}

			/** generate an signed 32bit integer in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline int32_t random(int32_t min_value, int32_t max_value)
			{
				return min_value + random_s32() % (max_value-min_value+1);
			}

			/** generate an 32bit floating point number in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline float32_t random(float32_t min_value, float32_t max_value)
			{
				return min_value + ((max_value-min_value) * random_close());
			}

			/** generate an 64bit floating point number in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline float64_t random(float64_t min_value, float64_t max_value)
			{
				return min_value + ((max_value-min_value) * random_close());
			}

			/** generate an 96-128bit floating point number (depending on the
			 * size of floatmax_t) in the range
			 * [min_value, max_value] (closed interval!)
			 *
			 * @param min_value minimum value
			 * @param max_value maximum value
			 * @return random number
			 */
			inline floatmax_t random(floatmax_t min_value, floatmax_t max_value)
			{
				return min_value + ((max_value-min_value) * random_close());
			}

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
			 *
			 * @param mu mean
			 * @param sigma variance
			 * @return sample from the desired normal distrib
			 */
			float64_t normal_distrib(float64_t mu, float64_t sigma) const;

			/**
			 * Sample a standard normal distribution,
			 * i.e. mean = 0, var = 1.0
			 *
			 * @return sample from the std normal distrib
			 */
			float64_t std_normal_distrib() const;

			/**
			 * Generate a seed for PRNG
			 *
			 * @return entropy for PRNG
			 */
			static uint32_t generate_seed();

			virtual const char* get_name() const { return "Random"; }

		private:
			/** initialise the object */
			void init();

			/** reinit PRNG
			 *
			 * @param seed seed for the PRNG
			 */
			 void reinit(uint32_t seed);

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

			/** seed */
			uint32_t m_seed;

			/** SFMT struct for 32-bit random */
			SFMT_T* m_sfmt_32;

			/** SFMT struct for 64-bit random */
			SFMT_T* m_sfmt_64;

			/** dSFMT struct */
			DSFMT_T* m_dsfmt;

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

			/** state lock */
			CLock m_state_lock;
	};
}

#endif /* __RANDOM_H__ */
