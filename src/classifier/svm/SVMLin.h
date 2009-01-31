/*
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3 of the License, or
 * (at your option) any later version.
 *
 * Written (W) 2006-2008 Soeren Sonnenburg
 * Copyright (C) 2006-2008 Fraunhofer Institute FIRST and Max-Planck-Society
 */

#ifndef _SVMLIN_H___
#define _SVMLIN_H___

#include "lib/common.h"
#include "classifier/LinearClassifier.h"
#include "features/DotFeatures.h"
#include "features/Labels.h"

/** class SVMLin */
class CSVMLin : public CLinearClassifier
{
	public:
		/** default constructor */
		CSVMLin();

		/** constructor
		 *
		 * @param C constant C
		 * @param traindat training features
		 * @param trainlab labels for features
		 */
		CSVMLin(
			float64_t C, CDotFeatures* traindat,
			CLabels* trainlab);
		virtual ~CSVMLin();

		/** get classifier type
		 *
		 * @return classifier type SVMLIN
		 */
		virtual inline EClassifierType get_classifier_type() { return CT_SVMLIN; }

		/** train classifier */
		virtual bool train();

		/** set C
		 *
		 * @param c1 new C1
		 * @param c2 new C2
		 */
		inline void set_C(float64_t c1, float64_t c2) { C1=c1; C2=c2; }

		/** get C1
		 *
		 * @return C1
		 */
		inline float64_t get_C1() { return C1; }

		/** get C2
		 *
		 * @return C2
		 */
		inline float64_t get_C2() { return C2; }

		/** set if bias shall be enabled
		 *
		 * @param enable_bias if bias shall be enabled
		 */
		inline void set_bias_enabled(bool enable_bias) { use_bias=enable_bias; }

		/** get if bias is enabled
		 *
		 * @return if bias is enabled
		 */
		inline bool get_bias_enabled() { return use_bias; }

		/** set epsilon
		 *
		 * @param eps new epsilon
		 */
		inline void set_epsilon(float64_t eps) { epsilon=eps; }

		/** get epsilon
		 *
		 * @return epsilon
		 */
		inline float64_t get_epsilon() { return epsilon; }

		/** @return object name */
		inline virtual const char* get_name() { return "SVMLin"; }

	protected:
		/** C1 */
		float64_t C1;
		/** C2 */
		float64_t C2;
		/** epsilon */
		float64_t epsilon;

		/** if bias is used */
		bool use_bias;
};
#endif
