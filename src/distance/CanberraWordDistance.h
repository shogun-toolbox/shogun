/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007 Christian Gehl
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CANBERRAWORDDISTANCE_H___
#define _CANBERRAWORDDISTANCE_H___

#include "lib/common.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "distance/StringDistance.h"

/** class CanberraWordDistance */
class CCanberraWordDistance: public CStringDistance<uint16_t>
{
	public:
		/** default constructor */
		CCanberraWordDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CCanberraWordDistance(CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r);
		virtual ~CCanberraWordDistance();

		/** init distance
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 * @return if init was successful
		 */
		virtual bool init(CFeatures* l, CFeatures* r);

		/** cleanup distance */
		virtual void cleanup();

		/** load init data from file
		 *
		 * @param src file to load from
		 * @return if loading was successful
		 */
		bool load_init(FILE* src);

		/** save init data to file
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		bool save_init(FILE* dest);

		/** get distance type we are
		 *
		 * @return distance type CHEBYSHEW
		 */
		virtual EDistanceType get_distance_type() { return D_CANBERRAWORD; }

		/** get name of the distance
		 *
		 * @return name Chebyshew-Metric
		 */
		virtual const char* get_name() { return "CanberraWord"; }

		/** get dictionary weights
		 *
		 * @param dsize size of the dictionary
		 * @param dweights dictionary weights are stored in here
		 */
		void get_dictionary(int32_t& dsize, float64_t*& dweights)
		{
			dsize=dictionary_size;
			dweights = dictionary_weights;
		}

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		float64_t compute(int32_t idx_a, int32_t idx_b);

	protected:
		/** size of the dictionary */
		int32_t dictionary_size;
		/** dictionary weights */
		float64_t* dictionary_weights;
};
#endif /* _CANBERRAWORDDISTANCE_H___ */
