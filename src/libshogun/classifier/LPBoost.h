/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2007-2008 Soeren Sonnenburg
 * Copyright (C) 2007-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _LPBOOST_H___
#define _LPBOOST_H___

#include "lib/config.h"
#ifdef USE_CPLEX

#include <stdio.h>
#include "lib/common.h"
#include "lib/DynamicArray.h"

#include "features/Features.h"
#include "features/SparseFeatures.h"
#include "classifier/LinearClassifier.h"

class CLPBoost : public CLinearClassifier
{
	public:
		CLPBoost();
		virtual ~CLPBoost();

		virtual bool train();

		inline virtual EClassifierType get_classifier_type()
		{
			return CT_LPBOOST;
		}

		bool init(int32_t num_vec);
		void cleanup();

		/** set features
		 *
		 * @param feat features to set
		 */
		virtual inline void set_features(CDotFeatures* feat)
		{
			if (feat->get_feature_class() != C_SPARSE ||
				feat->get_feature_type() != F_DREAL)
				SG_ERROR("LPBoost requires SPARSE REAL valued features\n");

			CLinearClassifier::set_features(feat);
		}

		inline void set_C(float64_t c1, float64_t c2) { C1=c1; C2=c2; }

		inline float64_t get_C1() { return C1; }
		inline float64_t get_C2() { return C2; }

		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }
		inline bool get_bias_enabled() { return use_bias; }

		inline void set_epsilon(float64_t eps) { epsilon=eps; }
		inline float64_t get_epsilon() { return epsilon; }

		float64_t find_max_violator(int32_t& max_dim);

		/** @return object name */
		inline virtual const char* get_name() const { return "LPBoost"; }

	protected:
		float64_t C1;
		float64_t C2;
		bool use_bias;
		float64_t epsilon;

		float64_t* u;
		CDynamicArray<int32_t>* dim;

		int32_t num_sfeat;
		int32_t num_svec;
		TSparse<float64_t>* sfeat;

};
#endif //USE_CPLEX
#endif //_LPBOOST_H___
