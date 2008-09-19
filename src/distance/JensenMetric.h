/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006 Christian Gehl
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _JENSENMETRIC_H___
#define _JENSENMETRIC_H___

#include "lib/common.h"
#include "distance/SimpleDistance.h"
#include "features/RealFeatures.h"

/** class JensenMetric 
 *
 * The Jensen-Shannon distance/divergence measures the similarity between
 * two data points which is based on the Kullback-Leibler divergence.
 *  
 * \f[\displaystyle
 *  d(\bf{x},\bf{x'}) = \sum_{i=0}^{n} x_{i} log\frac{x_{i}}{0.5(x_{i}
 *  +x'_{i})} + x'_{i} log\frac{x'_{i}}{0.5(x_{i}+x'_{i})}
 * \f]
 *
 * @see <a href="http://en.wikipedia.org/wiki/Jensen-Shannon_divergence">
 * Wikipedia: Jensen-Shannon divergence</a>
 * @see <a href="http://en.wikipedia.org/wiki/Kullback-Leibler_divergence">
 * Wikipedia: Kullback-Leibler divergence</a>             
 */
class CJensenMetric: public CSimpleDistance<DREAL>
{
	public:
		/** default constructor */
		CJensenMetric();

		/** constructor
		 *
		 * @param l features of left-hand side
		 * @param r features of right-hand side
		 */
		CJensenMetric(CRealFeatures* l, CRealFeatures* r);
		virtual ~CJensenMetric();

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
		 * @return distance type JENSEN
		 */
		virtual EDistanceType get_distance_type() { return D_JENSEN; }

		/** get name of the distance
		 *
		 * @return name Jensen-Metric
		 */
		virtual const CHAR* get_name() { return "Jensen-Metric"; };

	protected:
		/// compute distance for features a and b
		/// idx_{a,b} denote the index of the feature vectors
		/// in the corresponding feature object
		virtual DREAL compute(INT idx_a, INT idx_b);
};

#endif /* _JENSENMETRIC_H___ */
