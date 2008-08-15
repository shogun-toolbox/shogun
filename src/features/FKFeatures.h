/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 1999-2008 Soeren Sonnenburg
 * Written (W) 1999-2008 Gunnar Raetsch
 * Copyright (C) 1999-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _CFKFEATURES__H__
#define _CFKFEATURES__H__

#include "features/RealFeatures.h"
#include "distributions/hmm/HMM.h"

/** The class FKFeatures implements Fischer kernel features obtained from
 * two Hidden Markov models and was used in
 *
 * K. Tsuda, M. Kawanabe, G. Raetsch, S. Sonnenburg, and K.R. Mueller. A new
 * discriminative kernel from probabilistic models. Neural Computation,
 * 14:2397-2414, 2002.
 *
 * which also has the details.
 *
 * Note that FK-features are computed on the fly, so to be effective feature
 * caching should be enabled.
 *
 * It inherits its functionality from CSimpleFeatures, which should be
 * consulted for further reference.
 */
class CFKFeatures: public CRealFeatures
{
	public:
		/** constructor
		 *
		 * @param size cache size
		 * @param p positive HMM
		 * @param n negative HMM
		 */
		CFKFeatures(INT size, CHMM* p, CHMM* n);

		/** copy constructor */
		CFKFeatures(const CFKFeatures &orig);

		virtual ~CFKFeatures();

		/** set HMMs
		 *
		 * @param p positive HMM
		 * @param n negative HMM
		 */
		void set_models(CHMM* p, CHMM* n);

		/** set weight a
		 *
		 * @param a weight a
		 */
		inline void set_a(DREAL a)
		{
			weight_a=a;
		}

		/** get weight a
		 *
		 * @return weight a
		 */
		inline DREAL get_a()
		{
			return weight_a;
		}

		/** set feature matrix
		 *
		 * @return something floaty
		 */
		virtual DREAL* set_feature_matrix();

		/** set opt a
		 *
		 * @param a a
		 * @return something floaty
		 */
		double set_opt_a(double a=-1);

		/** get weight_a
		 *
		 * @return weight_a
		 */
		inline DREAL get_weight_a() { return weight_a; };

	protected:
		/** compute feature vector
		 *
		 * @param num num
		 * @param len len
		 * @param target
		 * @return something floaty
		 */
		virtual DREAL* compute_feature_vector(INT num, INT& len, DREAL* target=NULL);

		/** computes the feature vector to the address addr
		 *
		 * @param addr address
		 * @param num num
		 * @param len len
		 */
		void compute_feature_vector(DREAL* addr, INT num, INT& len);

		/** deriv a
		 *
		 * @param a a
		 * @param dimension dimension
		 */
		double deriv_a(double a, INT dimension=-1) ;

	protected:
		/** positive HMM */
		CHMM* pos;
		/** negative HMM */
		CHMM* neg;
		/** positive prob */
		double* pos_prob;
		/** negative prob */
		double* neg_prob;
		/** weight a */
		DREAL weight_a;
};
#endif
