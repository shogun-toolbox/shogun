/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2008 Christian Gehl
 * Copyright (C) 2008 Fraunhofer Institute FIRST
 */

#ifndef _COSINEDISTANCE_H___
#define _COSINEDISTANCE_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

/** class CosineDistance 
 *
 * The Cosine distance is obtained by using the Cosine similarity (Orchini
 * similarity, angular similarity, normalized dot product), which
 * measures similarity between two vectors by finding their angle.
 * An extension to the Cosine similarity yields the Tanimoto coefficient.
 * 
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = 1 - \frac{\sum_{i=1}^{n}\bf{x_{i}}\bf{x'_{i}}}
 *  {\sqrt{\sum_{i=1}^{n} x_{i}^2 \sum_{i=1}^{n} {x'}_{i}^2}} \quad x,x' \in R^{n}
 * \f] 
 * 
 * @see <a href="http://en.wikipedia.org/wiki/Cosine_similarity"> Wikipedia:
 * Cosine similarity </a>
 * @see CTanimotoDistance
 */
class CCosineDistance: public CSimpleDistance<DREAL>
{
	public:
		/** default constructor */
		CCosineDistance();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CCosineDistance(CRealFeatures* l, CRealFeatures* r);
		virtual ~CCosineDistance();

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
		 * @return distance type COSINE
		 */
		virtual EDistanceType get_distance_type() { return D_COSINE; }

		/** get name of the distance
		 *
		 * @return name Cosine distance
		 */
		virtual const char* get_name() { return "Cosine distance"; }

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(int32_t idx_a, int32_t idx_b);
};

#endif /* _COSINEDISTANCE_H___ */
