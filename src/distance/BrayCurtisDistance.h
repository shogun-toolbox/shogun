/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Christian Gehl
 * Copyright (C) 2008 Fraunhofer Institute FIRST
 */

#ifndef _BRAYCURTISDISTANCE_H___
#define _BRAYCURTISDISTANCE_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

/** class BrayCurtisDistance 
 *
 * The Bray-Curtis distance (Sorensen distance) is similar to the 
 * Manhattan distance with normalization.
 * 
 * \f[\displaystyle
 *  d(\bf{x},\bf{x}') = \frac{\sum_{i=1}^{n}|x_{i}-x'_{i}|}{\sum_{i=1}^{n}|x_{i}
 *  +x'_{i}|} \quad x,x' \in R^{n}
 *  \f]
 * 
 */
class CBrayCurtisDistance: public CSimpleDistance<DREAL>
{
	public:
		/** default constructor */
		CBrayCurtisDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CBrayCurtisDistance(CRealFeatures* l, CRealFeatures* r);
		virtual ~CBrayCurtisDistance();

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
		virtual bool load_init(FILE* src);

		/** save init data to file
		 *
		 * @param dest file to save to
		 * @return if saving was successful
		 */
		virtual bool save_init(FILE* dest);

		/** get distance type we are
		 *
		 * @return distance type BRAYCURTIS
		 */
		virtual EDistanceType get_distance_type() { return D_BRAYCURTIS; }

		/** get name of the distance
		 *
		 * @return name Bray-Curtis distance
		 */
		virtual const CHAR* get_name() { return "Bray-Curtis distance"; };

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
};

#endif /* _BRAYCURTISDISTANCE_H___ */
