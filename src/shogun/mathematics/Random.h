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

#define RAND_MAX_32 4294967296.0
#define RAND_MAX_64 18446744073709551616.0L

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

		private:
			/** seed */
			uint32_t m_seed;

			/** SFMT struct for 32-bit random */
			sfmt_t* m_sfmt_32;

			/** SFMT struct for 64-bit random */
			sfmt_t* m_sfmt_64;

			/** dSFMT struct */
			dsfmt_t* m_dsfmt;

#ifdef HAVE_PTHREAD
			/** state lock */
			PTHREAD_LOCK_T m_state_lock;
#endif

	};
}

#endif /* __RANDOM_H__ */
