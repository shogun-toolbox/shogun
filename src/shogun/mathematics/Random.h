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

			uint32_t random_32();
			
			uint64_t random_64();

			void fill_array_32(uint32_t* array, int size);

			void fill_array_64(uint64_t* array, int size);

			float64_t random_close();
			float64_t random_open();
			float64_t random_half_open();
			float64_t random_float_res53();

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

#ifdef HAVE_PTHREAD
			/** state lock */
			PTHREAD_LOCK_T m_state_lock;
#endif

	};
}

#endif /* __RANDOM_H__ */
