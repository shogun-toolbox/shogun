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

#ifndef _MANHATTANWORDDISTANCE_H___
#define _MANHATTANWORDDISTANCE_H___

#include "lib/common.h"
#include "features/Features.h"
#include "features/StringFeatures.h"
#include "distance/StringDistance.h"

/** class ManhattanWordDistance */
class CManhattanWordDistance: public CStringDistance<uint16_t>
{
	public:
		/** default constructor */
		CManhattanWordDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CManhattanWordDistance(CStringFeatures<uint16_t>* l, CStringFeatures<uint16_t>* r);
		virtual ~CManhattanWordDistance();

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
		 * @return distance type MANHATTANWORD
		 */
		virtual EDistanceType get_distance_type() { return D_MANHATTANWORD; }

		/** get name of the distance
		 *
		 * @return name ManhattanWord
		 */
		virtual const char* get_name() { return "ManhattanWord"; }

		/** get dictionary weights
		 *
		 * @param dsize size of the dictionary
		 * @param dweights dictionary weights are stored in here
		 */
		void get_dictionary(INT& dsize, DREAL*& dweights)
		{
			dsize=dictionary_size;
			dweights = dictionary_weights;
		}

	protected:
		/// compute distance function for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		DREAL compute(INT idx_a, INT idx_b);

	protected:
		/** size of the dictionary */
		INT dictionary_size;
		/** dictionary weights */
		DREAL* dictionary_weights;
};
#endif /* _MANHATTANWORDDISTANCE_H___ */
