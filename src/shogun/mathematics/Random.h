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

namespace shogun
{

#ifndef DOXYGEN_SHOULD_SKIP_THIS	
	enum PRNG_STATE
	{
		PRNG_32 = 1,
		PRNG_64 = 2
	};
#endif

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
			void seed(uint32_t seed);

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
			/** register parameters */
			void register_params();

			/** initialise the object */
			void init();

			/** reinit for new state
			 * 
			 * @param state PRNG_32 or PRNG_64
			 */
			void reinit(PRNG_STATE state);

		private:
			/** seed */
			uint32_t m_seed;

			/** SFMT struct */
			sfmt_t* m_sfmt;

			/** PRNG state */
			PRNG_STATE m_state;

#ifdef HAVE_PTHREAD
			/** state lock */
			PTHREAD_LOCK_T m_state_lock;
#endif

	};
}

#endif /* __RANDOM_H__ */
